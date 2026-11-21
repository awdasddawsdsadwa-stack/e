//! Forge — a Blender-inspired editor with an embedded MCP server.
//!
//! The main thread owns the window, scene and renderer. A dedicated tokio
//! thread runs the stdio MCP server and forwards tool calls here over channels.

mod capabilities;
mod ipc;
mod mesh_ops;
mod mcp;
mod render;
mod scene;
mod texture;
mod ui;

use capabilities::GpuProfile;
use ipc::AppCommand;
use render::Renderer;
use scene::{History, Material, MeshData, NodeId, OrbitCamera, PrimitiveKind, Scene, TextureData, Transform};
use serde_json::{json, Value};
use texture::{Pattern as TexturePattern, TextureOptions};
use three_d::*;
use ui::UiState;

fn main() {
    eprintln!("[forge] starting (OpenGL 3.3 Core via three-d 0.19, stdio MCP server)");

    let (command_tx, command_rx) = ipc::channel();
    let _mcp_thread = mcp::spawn(command_tx);

    let window = Window::new(WindowSettings {
        title: "Forge — 3D Editor + MCP".to_string(),
        initial_size: Some((1440, 900)),
        min_size: (800, 500),
        surface_settings: SurfaceSettings {
            // Old drivers handle MSAA poorly; keep the compatibility path safe.
            multisamples: 0,
            depth_buffer: 24,
            stencil_buffer: 0,
            hardware_acceleration: HardwareAcceleration::Preferred,
            vsync: true,
        },
        ..Default::default()
    })
    .expect("failed to create OpenGL 3.3 window — check your graphics driver");

    let context = window.gl();
    let profile = GpuProfile::detect(&context);
    let gpu_summary = profile.summary();
    let texture_limit = profile.texture_limit;
    eprintln!("[forge] OpenGL context created: {gpu_summary}");

    let mut scene = Scene::new();
    let mut history = History::default();
    let mut ui_state = UiState::default();
    let mut camera = OrbitCamera::default();
    let mut renderer = Renderer::new(&context, &scene);
    let mut gui = GUI::new(&context);

    let mut orbit_drag = false;
    let mut pan_drag = false;
    let mut last_pointer = (0.0f32, 0.0f32);
    let mut last_time = 0.0f64;

    window.render_loop(move |mut frame_input: FrameInput| {
        // 1. Drain commands queued by the MCP thread.
        while let Ok(command) = command_rx.try_recv() {
            handle_command(command, &mut scene, &mut history, texture_limit);
        }

        // 2. Advance the timeline when playing.
        if scene.timeline.playing {
            let seconds = (frame_input.accumulated_time - last_time) * 0.001;
            let advance = (seconds * scene.timeline.fps as f64).round() as i32;
            if advance > 0 {
                scene.timeline.frame += advance;
                if scene.timeline.frame > scene.timeline.frame_end {
                    scene.timeline.frame = scene.timeline.frame_start;
                }
            }
        }
        last_time = frame_input.accumulated_time;

        // 3. Build the UI first: three-d marks consumed input as `handled`.
        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |egui_ui| {
                ui::draw_ui(
                    egui_ui,
                    &mut scene,
                    &mut history,
                    &mut camera,
                    &mut ui_state,
                    true,
                    &gpu_summary,
                );
            },
        );

        // 4. Blender-style camera navigation on unhandled events.
        for event in &frame_input.events {
            match event {
                Event::MousePress { button, position, handled, .. } if !*handled => {
                    match button {
                        MouseButton::Left => orbit_drag = true,
                        MouseButton::Middle | MouseButton::Right => pan_drag = true,
                    }
                    last_pointer = (position.x, position.y);
                }
                Event::MouseRelease { handled, .. } if !*handled => {
                    orbit_drag = false;
                    pan_drag = false;
                }
                Event::MouseMotion { position, handled, .. } if !*handled => {
                    let dx = position.x - last_pointer.0;
                    let dy = position.y - last_pointer.1;
                    if orbit_drag {
                        camera.orbit(dx, dy);
                    } else if pan_drag {
                        camera.pan(dx, dy);
                    }
                    last_pointer = (position.x, position.y);
                }
                Event::MouseWheel { delta, handled, .. } if !*handled => {
                    camera.zoom(delta.1);
                }
                Event::KeyPress { kind, handled, .. } if !*handled => {
                    if *kind == Key::F {
                        if let Some(id) = scene.selected {
                            if let Some(node) = scene.node(id) {
                                let p = node.transform.position;
                                camera.focus(glam::Vec3::new(p[0], p[1], p[2]));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // 5. Sync GPU resources and draw.
        renderer.sync(&scene, ui_state.shading);

        let target = camera.target;
        let mut render_camera = Camera::new_perspective(
            frame_input.viewport,
            vec3(camera.position().x, camera.position().y, camera.position().z),
            vec3(target.x, target.y, target.z),
            vec3(0.0, 1.0, 0.0),
            degrees(45.0),
            0.1,
            1000.0,
        );
        renderer.render(
            &frame_input,
            &scene,
            &mut render_camera,
            ui_state.shading,
            ui_state.show_grid,
            &gui,
        );

        FrameOutput::default()
    });
}

fn handle_command(command: AppCommand, scene: &mut Scene, history: &mut History, texture_limit: u32) {
    eprintln!("[forge] MCP command #{}: {}", command.id, command.method);
    let result: Result<Value, String> = match command.method.as_str() {
        "get_scene_hierarchy" => Ok(scene.hierarchy_json()),

        "spawn_primitive" => {
            let kind_name = command
                .params
                .get("primitive")
                .and_then(Value::as_str)
                .unwrap_or("cube");
            match PrimitiveKind::parse(kind_name) {
                Some(kind) => {
                    let name = command
                        .params
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    let position = parse_array3(command.params.get("position"));
                    let color = parse_array4(command.params.get("color"));
                    history.push(scene);
                    let id = scene.spawn_primitive(kind, name, position, color);
                    Ok(json!({
                        "id": id,
                        "name": scene.node(id).map(|n| n.name.clone()).unwrap_or_default(),
                        "primitive": kind.as_str(),
                    }))
                }
                None => Err(format!("Unknown primitive '{kind_name}'. Use cube, sphere, cylinder, cone, plane or torus.")),
            }
        }

        "transform_object" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found (supply a valid 'id' or 'name')");
                return;
            };
            let position = parse_array3(command.params.get("position"));
            let scale = parse_array3(command.params.get("scale"));
            let rotation = if let Some(euler) = parse_array3(command.params.get("euler")) {
                let mut t = Transform::default();
                t.set_euler(euler[0], euler[1], euler[2]);
                Some(t.rotation)
            } else {
                parse_array4(command.params.get("rotation"))
            };
            history.push(scene);
            let ok = scene.transform_object(id, position, rotation, scale);
            if ok {
                Ok(json!({
                    "id": id,
                    "transform": scene.node(id).unwrap().transform.to_json(),
                }))
            } else {
                Err("Transform failed".into())
            }
        }

        "create_custom_mesh" => {
            let Some(positions) = parse_vec3_array(command.params.get("positions")) else {
                return_fail(command, "'positions' must be an array of [x, y, z] triples");
                return;
            };
            let Some(indices) = parse_u32_array(command.params.get("indices")) else {
                return_fail(command, "'indices' must be an array of vertex indices");
                return;
            };
            if positions.is_empty() || indices.is_empty() || indices.len() % 3 != 0 {
                return_fail(command, "positions must be non-empty and indices must be a multiple of 3");
                return;
            }
            let mut mesh = MeshData {
                positions,
                normals: parse_vec3_array(command.params.get("normals")).unwrap_or_default(),
                uvs: parse_vec2_array(command.params.get("uvs")).unwrap_or_default(),
                indices,
            };
            let auto_uv = command.params.get("auto_uv").and_then(Value::as_bool).unwrap_or(true);
            if auto_uv {
                mesh_ops::box_project_uvs(&mut mesh, 1.0);
            }
            if mesh.normals.len() != mesh.positions.len() {
                mesh.recalculate_normals();
            }
            if mesh.uvs.len() != mesh.positions.len() {
                mesh_ops::box_project_uvs(&mut mesh, 1.0);
            }
            let vertex_count = mesh.positions.len();
            let triangle_count = mesh.indices.len() / 3;
            let name = command.params.get("name").and_then(Value::as_str).map(str::to_string);
            let base_color = parse_array4(command.params.get("base_color"))
                .unwrap_or([0.72, 0.74, 0.78, 1.0]);
            history.push(scene);
            let id = scene.create_custom_mesh(
                name,
                mesh,
                Some(Material { base_color, ..Default::default() }),
            );
            Ok(json!({
                "id": id,
                "name": scene.node(id).map(|n| n.name.clone()).unwrap_or_default(),
                "vertices": vertex_count,
                "triangles": triangle_count,
                "auto_uv": auto_uv,
            }))
        }

        "modify_mesh" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found");
                return;
            };
            let op = command
                .params
                .get("operation")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let outcome: Result<(), String> = match op.as_str() {
                "extrude" => {
                    let direction = parse_array3(command.params.get("direction"))
                        .unwrap_or([0.0, 1.0, 0.0]);
                    let distance = command.params.get("distance").and_then(Value::as_f64)
                        .unwrap_or(1.0) as f32;
                    match scene.mesh_mut(id) {
                        Some(mesh) => mesh_ops::extrude(mesh, direction, distance),
                        None => Err("Node is not a mesh".into()),
                    }
                }
                "subdivide" => {
                    let levels = command.params.get("levels").and_then(Value::as_u64)
                        .unwrap_or(1).min(3) as u32;
                    match scene.mesh_mut(id) {
                        Some(mesh) => mesh_ops::subdivide(mesh, levels),
                        None => Err("Node is not a mesh".into()),
                    }
                }
                "merge" => {
                    let Some(source_id) = command.params.get("source_id").and_then(Value::as_u64)
                    else {
                        return_fail(command, "'source_id' is required for merge");
                        return;
                    };
                    let Some(source) = scene.mesh(source_id as NodeId).cloned() else {
                        return_fail(command, "source mesh not found");
                        return;
                    };
                    match scene.mesh_mut(id) {
                        Some(mesh) => {
                            mesh_ops::merge(mesh, &source);
                            Ok(())
                        }
                        None => Err("Target is not a mesh".into()),
                    }
                }
                "recalculate_normals" => match scene.mesh_mut(id) {
                    Some(mesh) => {
                        mesh_ops::recalculate_normals(mesh);
                        Ok(())
                    }
                    None => Err("Node is not a mesh".into()),
                },
                "delete_faces" => {
                    let mut faces = Vec::new();
                    if let Some(arr) = command.params.get("faces").and_then(Value::as_array) {
                        for value in arr {
                            if let Some(index) = value.as_u64() {
                                faces.push(index as usize);
                            }
                        }
                    }
                    match scene.mesh_mut(id) {
                        Some(mesh) => mesh_ops::delete_faces(mesh, &faces),
                        None => Err("Node is not a mesh".into()),
                    }
                }
                other => Err(format!("Unknown mesh operation: {other}")),
            };

            match outcome {
                Ok(()) => {
                    scene.revision += 1;
                    Ok(json!({
                        "id": id,
                        "operation": op,
                        "vertices": scene.mesh(id).map(|m| m.positions.len()).unwrap_or(0),
                        "triangles": scene.mesh(id).map(|m| m.indices.len() / 3).unwrap_or(0),
                    }))
                }
                Err(message) => Err(message),
            }
        }

        "apply_material" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found");
                return;
            };
            if scene.mesh(id).is_none() {
                return_fail(command, "Node is not a mesh");
                return;
            }
            let slot = command
                .params
                .get("slot")
                .and_then(Value::as_str)
                .unwrap_or("albedo")
                .to_string();
            let path = command.params.get("texture_path").and_then(Value::as_str).map(str::to_string);
            let loaded = match path {
                Some(path) => match texture::from_image_file(&path, None) {
                    Ok(mut tex) => {
                        texture::downscale(&mut tex, texture_limit);
                        Some(tex)
                    }
                    Err(message) => {
                        return_fail(command, &message);
                        return;
                    }
                },
                None => None,
            };
            history.push(scene);
            let mut summary = None;
            {
                if let Some(material) = scene.material_mut(id) {
                    if let Some(color) = parse_array4(command.params.get("base_color")) {
                        material.base_color = color;
                    }
                    if let Some(roughness) = command.params.get("roughness").and_then(Value::as_f64) {
                        material.roughness = roughness.clamp(0.0, 1.0) as f32;
                    }
                    if let Some(metallic) = command.params.get("metallic").and_then(Value::as_f64) {
                        material.metallic = metallic.clamp(0.0, 1.0) as f32;
                    }
                    if let Some(tex) = loaded {
                        assign_texture(material, &slot, tex);
                    }
                    summary = Some(material.summary_json());
                }
            }
            match summary {
                Some(value) => {
                    scene.revision += 1;
                    Ok(json!({
                        "id": id,
                        "name": scene.node(id).map(|n| n.name.clone()).unwrap_or_default(),
                        "material": value,
                    }))
                }
                None => Err("Node is not a mesh".into()),
            }
        }

        "generate_procedural_texture" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found");
                return;
            };
            if scene.mesh(id).is_none() {
                return_fail(command, "Node is not a mesh");
                return;
            }
            let pattern_name = command
                .params
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let pattern = match pattern_name.as_str() {
                "checkerboard" => TexturePattern::Checkerboard,
                "value_noise" => TexturePattern::ValueNoise,
                "wood" => TexturePattern::Wood,
                "brick" => TexturePattern::Brick,
                "gradient" => TexturePattern::Gradient,
                "scratches" => TexturePattern::Scratches,
                other => {
                    return_fail(command, &format!("Unknown pattern: {other}"));
                    return;
                }
            };
            let size = command.params.get("size").and_then(Value::as_u64)
                .unwrap_or(256).clamp(2, texture_limit as u64) as u32;
            let mut options = TextureOptions::default();
            if let Some(color) = parse_u8_color(command.params.get("color_a")) {
                options.color_a = color;
            }
            if let Some(color) = parse_u8_color(command.params.get("color_b")) {
                options.color_b = color;
            }
            if let Some(scale) = command.params.get("scale").and_then(Value::as_f64) {
                options.scale = scale as f32;
            }
            if let Some(seed) = command.params.get("seed").and_then(Value::as_u64) {
                options.seed = seed as u32;
            }
            let tex: TextureData = texture::generate(pattern, size, &options);
            let summary = tex.summary_json();
            let slot = command.params.get("slot").and_then(Value::as_str).unwrap_or("albedo");
            history.push(scene);
            let assigned = {
                if let Some(material) = scene.material_mut(id) {
                    assign_texture(material, slot, tex);
                    true
                } else {
                    false
                }
            };
            if assigned {
                scene.revision += 1;
                Ok(json!({
                    "id": id,
                    "name": scene.node(id).map(|n| n.name.clone()).unwrap_or_default(),
                    "slot": slot,
                    "pattern": pattern_name,
                    "size": size,
                    "color_a": options.color_a,
                    "color_b": options.color_b,
                    "scale": options.scale,
                    "seed": options.seed,
                    "texture": summary,
                }))
            } else {
                Err("Node is not a mesh".into())
            }
        }

        "undo" => {
            if history.undo(scene) {
                scene.revision += 1;
                Ok(json!({ "success": true, "message": "Undid last change" }))
            } else {
                Ok(json!({ "success": false, "message": "Nothing to undo" }))
            }
        }

        "redo" => {
            if history.redo(scene) {
                scene.revision += 1;
                Ok(json!({ "success": true, "message": "Redid last change" }))
            } else {
                Ok(json!({ "success": false, "message": "Nothing to redo" }))
            }
        }

        other => Err(format!("Unknown tool: {other}")),
    };

    let _ = command.reply.send(result);
}

fn return_fail(command: AppCommand, message: &str) {
    let _ = command.reply.send(Err(message.to_string()));
}

fn resolve_id(scene: &Scene, params: &Value) -> Option<NodeId> {
    if let Some(id) = params.get("id").and_then(Value::as_u64) {
        return Some(id as NodeId);
    }
    if let Some(name) = params.get("name").and_then(Value::as_str) {
        return scene.node_by_name(name).map(|n| n.id);
    }
    scene.selected
}

fn assign_texture(material: &mut Material, slot: &str, tex: TextureData) {
    match slot {
        "roughness" => material.roughness_map = Some(tex),
        "normal" => material.normal_map = Some(tex),
        _ => material.albedo = Some(tex),
    }
}

fn parse_vec3_array(value: Option<&Value>) -> Option<Vec<[f32; 3]>> {
    value?.as_array()?.iter().map(|v| parse_array3(Some(v))).collect()
}

fn parse_vec2_array(value: Option<&Value>) -> Option<Vec<[f32; 2]>> {
    let arr = value?.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let pair = item.as_array()?;
        if pair.len() != 2 {
            return None;
        }
        out.push([pair[0].as_f64()? as f32, pair[1].as_f64()? as f32]);
    }
    Some(out)
}

fn parse_u32_array(value: Option<&Value>) -> Option<Vec<u32>> {
    value?
        .as_array()?
        .iter()
        .map(|v| v.as_u64().map(|index| index as u32))
        .collect()
}

fn parse_u8_color(value: Option<&Value>) -> Option<[u8; 4]> {
    let arr = value?.as_array()?;
    if arr.len() != 4 {
        return None;
    }
    let mut out = [0u8; 4];
    for (slot, value) in arr.iter().enumerate() {
        out[slot] = value.as_u64()?.min(255) as u8;
    }
    Some(out)
}

fn parse_array3(value: Option<&Value>) -> Option<[f32; 3]> {
    let arr = value?.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    let mut out = [0.0f32; 3];
    for (i, v) in arr.iter().enumerate() {
        out[i] = v.as_f64()? as f32;
        if !out[i].is_finite() {
            return None;
        }
    }
    Some(out)
}

fn parse_array4(value: Option<&Value>) -> Option<[f32; 4]> {
    let arr = value?.as_array()?;
    if arr.len() != 4 {
        return None;
    }
    let mut out = [0.0f32; 4];
    for (i, v) in arr.iter().enumerate() {
        out[i] = v.as_f64()? as f32;
        if !out[i].is_finite() {
            return None;
        }
    }
    Some(out)
}

