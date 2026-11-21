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
            "description": "Return the full scene graph with every node's transform, material, geometry summary, visibility, parent, and WORLD-SPACE axis-aligned bounds (min/max/size for each mesh, plus an overall scene bounds). The timeline state is at both 'timeline' and 'scene.timeline'. Use the bounds to verify real 3D size and depth before declaring a model complete.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "transform_object",
            "description": "Move, rotate or scale a node (by id or case-insensitive name). Y is up; X/Z are ground. `position` sets the object's origin. `scale` multiplies the mesh's local vertex coordinates; use set_dimensions if you want to specify world size directly. `euler` is radians. Only supplied fields change; call get_object_info afterwards to confirm world-space bounds.",
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
            "description": "Render the scene to a PNG image block. You may set the camera explicitly with camera_position (eye) and camera_target (look-at), or use a view preset, or leave both unset to use the current editor camera. Y-up; useful for multi-angle self-checks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "width": { "type": "integer", "description": "Pixel width, clamped to 64..1024 (default 512)." },
                    "height": { "type": "integer", "description": "Pixel height, clamped to 64..1024 (default = width)." },
                    "view": {
                        "type": "string",
                        "enum": ["current", "front", "back", "left", "right", "top", "three_quarter"],
                        "description": "Optional camera preset (default current). front = +Z, right = +X, back = -Z, left = -X, top = +Y."
                    },
                    "camera_position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3, "description": "Explicit eye position [x,y,z] in world space (overrides view)." },
                    "camera_target": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3, "description": "Explicit look-at point [x,y,z] (defaults to the orbit target)." },
                    "yaw": { "type": "number", "description": "Orbit yaw in radians (alternative to camera_position)." },
                    "pitch": { "type": "number", "description": "Orbit pitch in radians (alternative to camera_position)." },
                    "distance": { "type": "number", "description": "Orbit distance (alternative to camera_position)." }
                },
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
            "name": "get_scene_info",
            "description": "Compact TEXT readback for measuring a scene: for each object reports world location, bounding-box size, ground contact (on ground / floating / below ground), rotation in DEGREES and scale. Use this to judge size and placement instead of guessing from a screenshot. Optional fields filter what is returned.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "items": { "type": "string", "enum": ["location", "size", "ground", "rotation", "scale", "material"] },
                        "description": "Which facts to include (default: all)."
                    },
                    "limit": { "type": "integer", "description": "Max objects to return (default 50)." }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "get_modeling_guide",
            "description": "On-demand guide for this editor: coordinate system, exact unit-primitive dimensions, how to compute scale from a desired size, ground-contact rules, and the self-check procedure for building solid 3D models.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        }),
        json!({
            "name": "set_dimensions",
            "description": "Set an object's world-space bounding-box size directly, e.g. size [4,0.6,1] for a 4x0.6x1 box. Avoids guessing scale multipliers. Computes scale from the mesh's local bounds; Y-up with X/Z ground. Returns the resulting bounds.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "size": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 3, "maxItems": 3,
                        "description": "Desired world-space extents [X, Y, Z]."
                    }
                },
                "required": ["size"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "get_object_info",
            "description": "Get full details for one node by id or name: transform, visibility, material, geometry statistics, light properties, and WORLD-Space bounds (min/max/size). Use bounds.size to check that a mesh has real depth on all axes.",
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
            "description": "Blender-style mesh modifiers: mirror across an axis, array copies, solidify (gives a flat mesh real thickness — use it on planes), smooth, triangulate, subdivide, extrude or recalculate normals. Check bounds after applying.",
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
            "name": "create_mesh_from_faces",
            "description": "BLENDER-STYLE TOPOLOGY: build a mesh from explicit vertices and polygon faces (like bmesh.from_pydata). Prefer this over stacking primitives for complex shapes. Faces are arrays of vertex indices with 3 or more entries; they are triangulated automatically, with smooth normals and box UVs generated. Y-up, coordinates are local to the object.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "vertices": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                        "description": "Local vertex positions [[x,y,z], ...]."
                    },
                    "faces": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "integer" } },
                        "description": "Polygon faces as vertex index lists, e.g. [[0,1,2,3]]."
                    },
                    "color": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "roughness": { "type": "number" },
                    "metallic": { "type": "number" }
                },
                "required": ["vertices", "faces"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "get_mesh_topology",
            "description": "Inspect a mesh's topology: vertex count, triangle count, edge count, and a list of triangles with their face indices and corner positions. Use face indices with extrude_faces/inset_faces.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "limit": { "type": "integer", "description": "Max triangles to list (default 100)." }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "extrude_faces",
            "description": "BLENDER-STYLE EDIT MODE: extrude selected triangles along their averaged normal, creating real side walls. Get face indices from get_mesh_topology.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "faces": { "type": "array", "items": { "type": "integer" } },
                    "distance": { "type": "number" }
                },
                "required": ["faces", "distance"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "inset_faces",
            "description": "BLENDER-STYLE EDIT MODE: inset selected triangles toward their centroids and rebuild the border faces. Get face indices from get_mesh_topology.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "faces": { "type": "array", "items": { "type": "integer" } },
                    "amount": { "type": "number" }
                },
                "required": ["faces", "amount"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "weld_vertices",
            "description": "Merge vertices closer than epsilon (merge-by-distance) and recompute normals. Useful after building faces from separate pieces.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "epsilon": { "type": "number" }
                },
                "required": ["epsilon"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "flip_normals",
            "description": "Reverse every triangle's winding and negate its normals — use when a mesh renders inside-out.",
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
            "name": "bevel_mesh",
            "description": "Approximate edge chamfer: pull each vertex toward its neighbours by amount and recompute normals. Gives a softer, low-poly bevel look.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" },
                    "name": { "type": "string" },
                    "amount": { "type": "number" }
                },
                "required": ["amount"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "create_armature",
            "description": "Create an armature (skeleton) object. Add bones with add_bone, then bind a mesh with auto_bind_skin.",
            "inputSchema": {
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "add_bone",
            "description": "Add a bone to an armature using bind-time world-space head and tail positions. Parent index (optional) makes a bone chain. Bone local +Y runs from head to tail.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "armature": { "type": "string", "description": "Armature node name (or use armature_id)." },
                    "armature_id": { "type": "integer" },
                    "name": { "type": "string" },
                    "parent": { "type": "integer", "description": "Parent bone index." },
                    "head": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "tail": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }
                },
                "required": ["name", "head", "tail"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "auto_bind_skin",
            "description": "Bind a mesh to an armature with automatic distance-envelope weights (up to 4 bones per vertex, normalized). Both mesh and armature should be at the origin with identity transforms when binding.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mesh": { "type": "string", "description": "Mesh node name (or use mesh_id)." },
                    "mesh_id": { "type": "integer" },
                    "armature": { "type": "string" },
                    "armature_id": { "type": "integer" },
                    "max_weights": { "type": "integer", "description": "Max bones per vertex, capped at 4 (default 4)." }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "set_bone_pose",
            "description": "Pose a bone by index with euler angles (radians), quaternion, and optional local translation. The mesh deforms immediately.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "armature": { "type": "string" },
                    "armature_id": { "type": "integer" },
                    "bone": { "type": "integer", "description": "Bone index." },
                    "euler": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }
                },
                "required": ["bone"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "get_armature_info",
            "description": "Inspect an armature: bone names, parents, bind-time head/tail, current pose rotations, and which meshes are bound to it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "armature": { "type": "string" },
                    "armature_id": { "type": "integer" }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "reset_pose",
            "description": "Reset every bone in an armature to its rest pose.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "armature": { "type": "string" },
                    "armature_id": { "type": "integer" }
                },
                "additionalProperties": false
            }
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
        "transform_object",
        "modify_mesh",
        "apply_material",
        "generate_procedural_texture",
        "render_viewport_snapshot",
        "clear_scene",
        "duplicate_node",
        "set_camera",
        "create_mesh_from_faces",
        "get_mesh_topology",
        "extrude_faces",
        "inset_faces",
        "weld_vertices",
        "flip_normals",
        "bevel_mesh",
        "create_armature",
        "add_bone",
        "auto_bind_skin",
        "set_bone_pose",
        "get_armature_info",
        "reset_pose",
        "get_scene_info",
        "get_modeling_guide",
        "set_dimensions",
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
        assert!(names.contains(&"create_mesh_from_faces"));
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
