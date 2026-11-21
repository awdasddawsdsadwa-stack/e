//! Model Context Protocol server over stdio (JSON-RPC 2.0).
//!
//! Runs on its own tokio runtime in a dedicated thread. stdout is reserved
//! exclusively for newline-delimited JSON-RPC messages.

use crate::ipc::{AppCommand, CommandSender};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::oneshot;

const PROTOCOL_VERSION: &str = "2025-06-18";
const SUPPORTED_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
const SERVER_NAME: &str = "forge-mcp";
const SERVER_VERSION: &str = "0.1.0";

static NEXT_COMMAND_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

/// Spawn the blocking tokio runtime + stdio loop on a dedicated OS thread.
pub fn spawn(command_tx: CommandSender) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("forge-mcp-stdio".into())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(err) => {
                    eprintln!("[forge-mcp] failed to build tokio runtime: {err}");
                    return;
                }
            };
            runtime.block_on(serve(command_tx));
        })
        .expect("failed to spawn MCP thread")
}

async fn serve(command_tx: CommandSender) {
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut stdout = tokio::io::stdout();
    let mut line = String::new();

    loop {
        line.clear();
        let read = match reader.read_line(&mut line).await {
            Ok(n) => n,
            Err(err) => {
                eprintln!("[forge-mcp] stdin read error: {err}");
                break;
            }
        };
        if read == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let Some(response) = dispatch(trimmed, &command_tx).await else {
            continue;
        };

        let mut bytes = match serde_json::to_vec(&response) {
            Ok(b) => b,
            Err(err) => {
                eprintln!("[forge-mcp] failed to serialise response: {err}");
                continue;
            }
        };
        bytes.push(b'\n');
        if let Err(err) = stdout.write_all(&bytes).await {
            eprintln!("[forge-mcp] stdout write error: {err}");
            break;
        }
        let _ = stdout.flush().await;
    }
    eprintln!("[forge-mcp] stdio loop ended");
}

async fn dispatch(raw: &str, tx: &CommandSender) -> Option<Value> {
    let request: RpcRequest = match serde_json::from_str(raw) {
        Ok(r) => r,
        Err(err) => {
            return Some(error_response(
                Value::Null,
                -32700,
                format!("Parse error: {err}"),
            ));
        }
    };

    if request.jsonrpc != "2.0" {
        return Some(error_response(
            request.id.clone().unwrap_or(Value::Null),
            -32600,
            "Invalid Request: jsonrpc must be '2.0'".into(),
        ));
    }

    // Notifications (no id) never receive a response.
    if request.id.is_none() {
        // `notifications/initialized` and anything else are intentionally ignored.
        return None;
    }
    let id = request.id.clone().unwrap();

    let params = request.params.clone().unwrap_or_else(|| json!({}));
    let result = match request.method.as_str() {
        "initialize" => {
            let requested = params.get("protocolVersion").and_then(Value::as_str);
            Ok(initialize_result(requested))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => call_tool(params, tx).await,
        other => Err((-32601, format!("Method not found: {other}"))),
    };

    Some(match result {
        Ok(value) => json!({ "jsonrpc": "2.0", "id": id, "result": value }),
        Err((code, message)) => error_response(id, code, message),
    })
}

fn error_response(id: Value, code: i64, message: String) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    })
}

fn initialize_result(requested: Option<&str>) -> Value {
    let negotiated = requested
        .filter(|v| SUPPORTED_VERSIONS.contains(v))
        .unwrap_or(PROTOCOL_VERSION);
    json!({
        "protocolVersion": negotiated,
        "capabilities": {
            "tools": { "listChanged": false }
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        },
        "instructions": "Forge exposes scene inspection and modeling tools. Call tools/list to discover them."
    })
}

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "get_scene_hierarchy",
            "description": "Return the full scene graph: every node with its transform, material, geometry summary, visibility and parent. The current timeline state is available both at the top-level 'timeline' key and at 'scene.timeline'. Use this to verify what is in the scene.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "spawn_primitive",
            "description": "Spawn a procedural primitive mesh into the scene and select it. The new node becomes the current selection, so later tools may omit id/name to target it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "primitive": {
                        "type": "string",
                        "enum": ["cube", "sphere", "cylinder", "cone", "plane", "torus"],
                        "description": "Which primitive to create."
                    },
                    "name": { "type": "string", "description": "Optional object name (auto-generated when omitted)." },
                    "position": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 3, "maxItems": 3,
                        "description": "World position [x, y, z]."
                    },
                    "color": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 4, "maxItems": 4,
                        "description": "Linear RGBA base colour."
                    }
                },
                "required": ["primitive"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "transform_object",
            "description": "Move, rotate or scale an existing node. Identify the node by numeric id or by exact (case-insensitive) name. Only the supplied fields are changed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer", "description": "Node id, as returned by get_scene_hierarchy." },
                    "name": { "type": "string", "description": "Node name (alternative to id)." },
                    "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "rotation": {
                        "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4,
                        "description": "Quaternion [x, y, z, w]."
                    },
                    "euler": {
                        "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3,
                        "description": "Euler angles in radians [x, y, z] (alternative to rotation)."
                    },
                    "scale": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "create_custom_mesh",
            "description": "Create a mesh from raw vertex buffers. Supply positions ([[x,y,z], ...]) and triangle indices (flat list of vertex indices, length divisible by 3). Normals and UVs are optional. With auto_uv=true, box-projection UVs are generated automatically.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "positions": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }
                    },
                    "normals": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }
                    },
                    "uvs": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 }
                    },
                    "indices": { "type": "array", "items": { "type": "integer" } },
                    "auto_uv": { "type": "boolean", "description": "Generate box-projection UVs (default true)." },
                    "base_color": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 }
                },
                "required": ["positions", "indices"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "modify_mesh",
            "description": "Apply a modeling operation to a mesh: extrude along an axis, subdivide triangles, merge another mesh, recalculate smooth normals, or delete triangles.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "operation": {
                        "type": "string",
                        "enum": ["extrude", "subdivide", "merge", "recalculate_normals", "delete_faces"]
                    },
                    "direction": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "distance": { "type": "number" },
                    "levels": { "type": "integer", "description": "Subdivision levels (1-3)." },
                    "source_id": { "type": "integer", "description": "Mesh to merge into the target." },
                    "faces": { "type": "array", "items": { "type": "integer" }, "description": "Triangle indices to delete." }
                },
                "required": ["operation"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "apply_material",
            "description": "Set PBR properties on a mesh and optionally load a PNG/JPG texture from disk. Textures are downscaled to the GPU-safe limit before upload. Target by id or name; selection is unchanged.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "base_color": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "roughness": { "type": "number" },
                    "metallic": { "type": "number" },
                    "texture_path": { "type": "string", "description": "Path to a .png/.jpg image." },
                    "slot": {
                        "type": "string",
                        "enum": ["albedo", "roughness", "normal"],
                        "description": "Which texture slot to write (default albedo)."
                    }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "generate_procedural_texture",
            "description": "Generate a deterministic RGBA8 texture (checkerboard, value noise, wood, brick, gradient or metallic scratches) and assign it to a mesh texture slot. Target by id or name; selection is unchanged. The response echoes all generation arguments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "pattern": {
                        "type": "string",
                        "enum": ["checkerboard", "value_noise", "wood", "brick", "gradient", "scratches"]
                    },
                    "size": { "type": "integer", "description": "Square size in pixels; clamped to the GPU-safe limit." },
                    "color_a": { "type": "array", "items": { "type": "integer" }, "minItems": 4, "maxItems": 4 },
                    "color_b": { "type": "array", "items": { "type": "integer" }, "minItems": 4, "maxItems": 4 },
                    "scale": { "type": "number" },
                    "seed": { "type": "integer" },
                    "slot": { "type": "string", "enum": ["albedo", "roughness", "normal"] }
                },
                "required": ["pattern"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "render_viewport_snapshot",
            "description": "Render the current scene from the editor camera and return a resized PNG as an image content block (base64). Use this to visually inspect your own work and catch mistakes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "width": { "type": "integer", "description": "Pixel width, clamped to 64..1024 (default 512)." },
                    "height": { "type": "integer", "description": "Pixel height, clamped to 64..1024 (default = width)." }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "build_model",
            "description": "Build a whole model in ONE call from a list of parts. Prefer this over many separate spawn calls: it is one undo step and much faster. Each part may include a procedural texture.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "parts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "primitive": { "type": "string", "enum": ["cube", "sphere", "cylinder", "cone", "plane", "torus"] },
                                "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                                "scale": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                                "euler": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                                "color": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                                "roughness": { "type": "number" },
                                "metallic": { "type": "number" },
                                "texture": {
                                    "type": "object",
                                    "properties": {
                                        "pattern": { "type": "string", "enum": ["checkerboard", "value_noise", "wood", "brick", "gradient", "scratches"] },
                                        "size": { "type": "integer" },
                                        "color_a": { "type": "array", "items": { "type": "integer" }, "minItems": 4, "maxItems": 4 },
                                        "color_b": { "type": "array", "items": { "type": "integer" }, "minItems": 4, "maxItems": 4 },
                                        "scale": { "type": "number" },
                                        "seed": { "type": "integer" },
                                        "slot": { "type": "string", "enum": ["albedo", "roughness", "normal"] }
                                    },
                                    "required": ["pattern"]
                                }
                            },
                            "required": ["primitive"]
                        }
                    }
                },
                "required": ["parts"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "clear_scene",
            "description": "Remove every mesh from the scene (camera and lights are kept). Useful before building a fresh model. This is undoable.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        }),
        json!({
            "name": "duplicate_node",
            "description": "Copy a mesh node, including its geometry and material, with an optional position offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string", "description": "Node to duplicate." },
                    "new_name": { "type": "string" },
                    "offset": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "set_camera",
            "description": "Frame the scene for a screenshot: set the orbit target, distance and angles (radians).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "distance": { "type": "number" },
                    "yaw": { "type": "number" },
                    "pitch": { "type": "number" }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "get_object_info",
            "description": "Get full details for one node by id or name: transform, visibility, material, geometry statistics and light properties.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "delete_object",
            "description": "Delete a node from the scene. Undoable.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "rename_object",
            "description": "Rename a node.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "new_name": { "type": "string" }
                },
                "required": ["new_name"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "set_object_visibility",
            "description": "Show or hide a node.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "visible": { "type": "boolean" }
                },
                "required": ["visible"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "select_object",
            "description": "Select a node so later tools can omit id/name and so it is highlighted in the viewport.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "apply_modifier",
            "description": "Blender-style mesh modifiers: mirror across an axis, array copies, solidify, smooth, triangulate, subdivide, extrude or recalculate normals.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "operation": {
                        "type": "string",
                        "enum": ["mirror", "array", "solidify", "smooth", "triangulate", "subdivide", "extrude", "recalculate_normals"]
                    },
                    "axis": { "type": "integer", "description": "mirror axis: 0=X, 1=Y, 2=Z" },
                    "count": { "type": "integer", "description": "array copy count" },
                    "offset": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "thickness": { "type": "number" },
                    "factor": { "type": "number" },
                    "iterations": { "type": "integer" },
                    "direction": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "distance": { "type": "number" },
                    "levels": { "type": "integer" }
                },
                "required": ["operation"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "set_material_preset",
            "description": "Apply a named PBR material preset (plastic, matte, metal, chrome, glass, rubber, gold, ceramic, emissive).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "preset": {
                        "type": "string",
                        "enum": ["plastic", "matte", "metal", "chrome", "glass", "rubber", "gold", "ceramic", "emissive"]
                    },
                    "color": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 }
                },
                "required": ["preset"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "create_light",
            "description": "Add a directional, point or ambient light to the scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "light_type": { "type": "string", "enum": ["directional", "point", "ambient"] },
                    "name": { "type": "string" },
                    "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "color": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "intensity": { "type": "number" }
                },
                "required": ["light_type"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "set_light",
            "description": "Change a light's colour, intensity or position.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "color": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "intensity": { "type": "number" },
                    "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "get_viewport_info",
            "description": "Return editor state: camera orbit values, shading mode, grid toggle, detected GPU profile and scene counts.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        }),
        json!({
            "name": "focus_selection",
            "description": "Move the orbit camera target to the selected object so the next screenshot frames it.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        }),
        json!({
            "name": "undo",
            "description": "Revert the most recent scene mutation.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        }),
        json!({
            "name": "redo",
            "description": "Re-apply the most recently undone scene mutation.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        }),
    ]
}

async fn call_tool(params: Value, tx: &CommandSender) -> Result<Value, (i64, String)> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| (-32602, "tools/call requires a 'name' field".to_string()))?
        .to_string();

    let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
    if !arguments.is_object() {
        return Err((-32602, "tool arguments must be an object".into()));
    }

    let known = [
        "get_scene_hierarchy",
        "spawn_primitive",
        "transform_object",
        "create_custom_mesh",
        "modify_mesh",
        "apply_material",
        "generate_procedural_texture",
        "render_viewport_snapshot",
        "build_model",
        "clear_scene",
        "duplicate_node",
        "set_camera",
        "get_object_info",
        "delete_object",
        "rename_object",
        "set_object_visibility",
        "select_object",
        "apply_modifier",
        "set_material_preset",
        "create_light",
        "set_light",
        "get_viewport_info",
        "focus_selection",
        "undo",
        "redo",
    ];
    if !known.contains(&name.as_str()) {
        return Ok(tool_error(&format!("Unknown tool: {name}")));
    }

    let cmd_id = NEXT_COMMAND_ID.fetch_add(1, Ordering::Relaxed);
    let (reply_tx, reply_rx) = oneshot::channel();
    let command = AppCommand {
        id: cmd_id,
        method: name.clone(),
        params: arguments,
        reply: reply_tx,
    };

    if tx.send(command).is_err() {
        return Ok(tool_error("Render thread is not available (application closed?)"));
    }

    match reply_rx.await {
        Ok(Ok(value)) => {
            // Tools that already return an MCP content array (e.g. screenshots)
            // are passed through so image blocks survive unchanged.
            if value.get("content").is_some() {
                Ok(value)
            } else {
                Ok(json!({
                    "content": [{ "type": "text", "text": serialize_text(&value) }],
                    "structuredContent": value,
                    "isError": false
                }))
            }
        }
        Ok(Err(message)) => Ok(tool_error(&message)),
        Err(_) => Ok(tool_error("Render thread dropped the request")),
    }
}

fn tool_error(message: &str) -> Value {
    json!({
        "content": [{ "type": "text", "text": message }],
        "isError": true
    })
}

fn serialize_text(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::channel;

    fn tx() -> CommandSender {
        channel().0
    }

    #[tokio::test]
    async fn initialize_negotiates_supported_version() {
        let raw = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#;
        let response = dispatch(raw, &tx()).await.unwrap();
        assert_eq!(response["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(response["result"]["serverInfo"]["name"], "forge-mcp");
        assert_eq!(response["result"]["capabilities"]["tools"]["listChanged"], false);
    }

    #[tokio::test]
    async fn tools_list_advertises_phase_one_tools() {
        let response = dispatch(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#, &tx())
            .await
            .unwrap();
        let names: Vec<&str> = response["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"get_scene_hierarchy"));
        assert!(names.contains(&"spawn_primitive"));
        assert!(names.contains(&"transform_object"));
        assert!(names.contains(&"undo"));
        assert!(names.contains(&"redo"));
    }

    #[tokio::test]
    async fn notifications_get_no_response() {
        let response = dispatch(
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            &tx(),
        )
        .await;
        assert!(response.is_none());
    }

    #[tokio::test]
    async fn unknown_method_returns_standard_error() {
        let response = dispatch(r#"{"jsonrpc":"2.0","id":3,"method":"nope"}"#, &tx())
            .await
            .unwrap();
        assert_eq!(response["error"]["code"], -32601);
    }
}
