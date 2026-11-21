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

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use capabilities::GpuProfile;
use ipc::AppCommand;
use render::Renderer;
use scene::{History, LightType, Material, MeshData, NodeId, NodeKind, OrbitCamera, PrimitiveKind, Scene, TextureData, Transform};
use serde_json::{json, Value};
use texture::{Pattern as TexturePattern, TextureOptions};
use std::collections::HashSet;
use three_d::*;
use tokio::sync::oneshot;
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
    let mut pending_capture: Option<(u32, u32, oneshot::Sender<Result<Value, String>>)> = None;

    window.render_loop(move |mut frame_input: FrameInput| {
        // 1. Drain commands queued by the MCP thread. Camera and screenshot
        // commands need main-thread state that is not part of `Scene`.
        while let Ok(command) = command_rx.try_recv() {
            match command.method.as_str() {
                "set_camera" => handle_camera_command(command, &mut camera),
                "focus_selection" => {
                    if let Some(id) = scene.selected {
                        if let Some(node) = scene.node(id) {
                            let p = node.transform.position;
                            camera.focus(glam::Vec3::new(p[0], p[1], p[2]));
                        }
                    }
                    let _ = command.reply.send(Ok(json!({
                        "target": camera.target.to_array(),
                        "selected": scene.selected,
                    })));
                }
                "get_viewport_info" => {
                    let _ = command.reply.send(Ok(json!({
                        "camera": {
                            "target": camera.target.to_array(),
                            "distance": camera.distance,
                            "yaw": camera.yaw,
                            "pitch": camera.pitch,
                            "position": camera.position().to_array(),
                        },
                        "editor": {
                            "shading": ui_state.shading.label(),
                            "mode": ui_state.mode.label(),
                            "show_grid": ui_state.show_grid,
                            "status": ui_state.status,
                        },
                        "gpu": gpu_summary,
                        "scene": {
                            "nodes": scene.nodes.len(),
                            "selected": scene.selected,
                            "can_undo": history.can_undo(),
                            "can_redo": history.can_redo(),
                        },
                    })));
                }
                "render_viewport_snapshot" => {
                    let width = command
                        .params
                        .get("width")
                        .and_then(Value::as_u64)
                        .unwrap_or(512)
                        .clamp(64, 1024) as u32;
                    let height = command
                        .params
                        .get("height")
                        .and_then(Value::as_u64)
                        .unwrap_or(width as u64)
                        .clamp(64, 1024) as u32;
                    let view = command
                        .params
                        .get("view")
                        .and_then(Value::as_str)
                        .unwrap_or("current")
                        .to_string();
                    match view.as_str() {
                        // Yaw places the camera around the target: 0 => +X, PI/2 => +Z.
                        "front" => {
                            camera.yaw = std::f32::consts::FRAC_PI_2;
                            camera.pitch = 0.0;
                        }
                        "back" => {
                            camera.yaw = -std::f32::consts::FRAC_PI_2;
                            camera.pitch = 0.0;
                        }
                        "right" => {
                            camera.yaw = 0.0;
                            camera.pitch = 0.0;
                        }
                        "left" => {
                            camera.yaw = std::f32::consts::PI;
                            camera.pitch = 0.0;
                        }
                        "top" => {
                            camera.yaw = 0.0;
                            camera.pitch = 1.45;
                        }
                        "three_quarter" => {
                            camera.yaw = -0.9;
                            camera.pitch = 0.35;
                        }
                        _ => {}
                    }
                    pending_capture = Some((width, height, command.reply));
                }
                _ => handle_command(command, &mut scene, &mut history, texture_limit),
            }
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

        // Reply to screenshot requests after the frame has been drawn.
        if let Some((width, height, reply)) = pending_capture.take() {
            let eye = camera.position();
            let look = camera.target;
            let response = match renderer.snapshot_png(
                &scene,
                vec3(eye.x, eye.y, eye.z),
                vec3(look.x, look.y, look.z),
                ui_state.shading,
                width,
                height,
            ) {
                Ok(bytes) => {
                    let encoded = STANDARD.encode(&bytes);
                    Ok(json!({
                        "content": [{
                            "type": "image",
                            "data": encoded,
                            "mimeType": "image/png"
                        }],
                        "structuredContent": {
                            "width": width,
                            "height": height,
                            "bytes": bytes.len(),
                            "shading": ui_state.shading.label(),
                        },
                        "isError": false,
                    }))
                }
                Err(message) => Ok(json!({
                    "content": [{ "type": "text", "text": message }],
                    "isError": true,
                })),
            };
            let _ = reply.send(response);
        }

        FrameOutput::default()
    });
}

fn handle_camera_command(command: AppCommand, camera: &mut OrbitCamera) {
    if let Some(target) = parse_array3(command.params.get("target")) {
        camera.focus(glam::Vec3::from(target));
    }
    if let Some(distance) = command.params.get("distance").and_then(Value::as_f64) {
        camera.distance = (distance as f32).clamp(0.5, 500.0);
    }
    if let Some(yaw) = command.params.get("yaw").and_then(Value::as_f64) {
        camera.yaw = yaw as f32;
    }
    if let Some(pitch) = command.params.get("pitch").and_then(Value::as_f64) {
        camera.pitch = (pitch as f32).clamp(-1.5, 1.5);
    }
    let _ = command.reply.send(Ok(json!({
        "target": camera.target.to_array(),
        "distance": camera.distance,
        "yaw": camera.yaw,
        "pitch": camera.pitch,
    })));
}

fn build_model(
    command: &AppCommand,
    scene: &mut Scene,
    history: &mut History,
    texture_limit: u32,
) -> Result<Value, String> {
    let parts = command
        .params
        .get("parts")
        .and_then(Value::as_array)
        .ok_or_else(|| "'parts' must be an array".to_string())?;
    history.push(scene);
    let mut built = Vec::new();

    for (index, part) in parts.iter().enumerate() {
        let kind_name = part.get("primitive").and_then(Value::as_str).unwrap_or("cube");
        let kind = PrimitiveKind::parse(kind_name)
            .ok_or_else(|| format!("invalid primitive at part {index}: {kind_name}"))?;
        let name = part.get("name").and_then(Value::as_str).map(str::to_string);
        let id = scene.spawn_primitive(
            kind,
            name,
            parse_array3(part.get("position")),
            parse_array4(part.get("color")),
        );

        if let Some(scale) = parse_array3(part.get("scale")) {
            scene.transform_object(id, None, None, Some(scale));
        }
        if let Some(euler) = parse_array3(part.get("euler")) {
            let mut transform = Transform::default();
            transform.set_euler(euler[0], euler[1], euler[2]);
            scene.transform_object(id, None, Some(transform.rotation), None);
        }
        {
            if let Some(material) = scene.material_mut(id) {
                if let Some(roughness) = part.get("roughness").and_then(Value::as_f64) {
                    material.roughness = roughness.clamp(0.0, 1.0) as f32;
                }
                if let Some(metallic) = part.get("metallic").and_then(Value::as_f64) {
                    material.metallic = metallic.clamp(0.0, 1.0) as f32;
                }
            }
        }
        if let Some(texture_config) = part.get("texture") {
            let pattern_name = texture_config
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or("checkerboard");
            let pattern = match pattern_name {
                "checkerboard" => TexturePattern::Checkerboard,
                "value_noise" => TexturePattern::ValueNoise,
                "wood" => TexturePattern::Wood,
                "brick" => TexturePattern::Brick,
                "gradient" => TexturePattern::Gradient,
                "scratches" => TexturePattern::Scratches,
                other => return Err(format!("unknown texture pattern at part {index}: {other}")),
            };
            let size = texture_config
                .get("size")
                .and_then(Value::as_u64)
                .unwrap_or(256)
                .clamp(2, texture_limit as u64) as u32;
            let mut options = TextureOptions::default();
            if let Some(color) = parse_u8_color(texture_config.get("color_a")) {
                options.color_a = color;
            }
            if let Some(color) = parse_u8_color(texture_config.get("color_b")) {
                options.color_b = color;
            }
            if let Some(scale) = texture_config.get("scale").and_then(Value::as_f64) {
                options.scale = scale as f32;
            }
            if let Some(seed) = texture_config.get("seed").and_then(Value::as_u64) {
                options.seed = seed as u32;
            }
            let tex = texture::generate(pattern, size, &options);
            let slot = texture_config.get("slot").and_then(Value::as_str).unwrap_or("albedo");
            if let Some(material) = scene.material_mut(id) {
                assign_texture(material, slot, tex);
            }
        }

        built.push(json!({
            "id": id,
            "name": scene.node(id).map(|n| n.name.clone()).unwrap_or_default(),
            "bounds": scene.node(id).and_then(|n| n.world_bounds().map(|(_, _, size)| size)),
        }));
    }

    let scene_bounds = {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for node in &scene.nodes {
            if let Some((lo, hi, _)) = node.world_bounds() {
                for axis in 0..3 {
                    min[axis] = min[axis].min(lo[axis]);
                    max[axis] = max[axis].max(hi[axis]);
                }
            }
        }
        json!({ "min": min, "max": max, "size": [max[0]-min[0], max[1]-min[1], max[2]-min[2]] })
    };
    scene.revision += 1;
    Ok(json!({ "count": built.len(), "parts": built, "scene_bounds": scene_bounds }))
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
                        "bounds": scene.node(id).and_then(|n| n.world_bounds().map(|(_, _, size)| size)),
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
                "bounds": scene.node(id).and_then(|n| n.world_bounds().map(|(_, _, size)| size)),
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
                        "bounds": scene.node(id).and_then(|n| n.world_bounds().map(|(_, _, size)| size)),
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

        "build_model" => build_model(&command, scene, history, texture_limit),

        "clear_scene" => {
            history.push(scene);
            let before = scene.nodes.len();
            scene.nodes.retain(|node| !matches!(node.kind, NodeKind::Mesh { .. }));
            scene.selected = None;
            scene.revision += 1;
            Ok(json!({ "removed": before - scene.nodes.len(), "remaining": scene.nodes.len() }))
        }

        "duplicate_node" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found");
                return;
            };
            let Some(source) = scene.node(id).cloned() else {
                return_fail(command, "Node not found");
                return;
            };
            let (mesh, material) = match &source.kind {
                NodeKind::Mesh { mesh, material, .. } => (mesh.clone(), material.clone()),
                _ => {
                    return_fail(command, "Only mesh nodes can be duplicated");
                    return;
                }
            };
            let new_name = command
                .params
                .get("new_name")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("{}_copy", source.name));
            let offset = parse_array3(command.params.get("offset")).unwrap_or([1.0, 0.0, 0.0]);
            history.push(scene);
            let copy_id = scene.create_custom_mesh(Some(new_name), mesh, Some(material));
            let p = source.transform.position;
            scene.transform_object(
                copy_id,
                Some([p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]]),
                Some(source.transform.rotation),
                Some(source.transform.scale),
            );
            Ok(json!({
                "id": copy_id,
                "name": scene.node(copy_id).map(|n| n.name.clone()).unwrap_or_default(),
                "source": id,
            }))
        }

        "create_mesh_from_faces" => {
            let Some(vertices) = parse_vec3_array(command.params.get("vertices")) else {
                return_fail(command, "'vertices' must be an array of [x,y,z]");
                return;
            };
            let Some(faces) = parse_faces(command.params.get("faces")) else {
                return_fail(command, "'faces' must be an array of index arrays");
                return;
            };
            let mesh = match mesh_ops::from_faces(&vertices, &faces) {
                Ok(mesh) => mesh,
                Err(message) => {
                    return_fail(command, &message);
                    return;
                }
            };
            let vertex_count = mesh.positions.len();
            let triangle_count = mesh.indices.len() / 3;
            let name = command.params.get("name").and_then(Value::as_str).map(str::to_string);
            let material = Material {
                base_color: parse_array4(command.params.get("color")).unwrap_or([0.72, 0.74, 0.78, 1.0]),
                roughness: command.params.get("roughness").and_then(Value::as_f64).unwrap_or(0.75) as f32,
                metallic: command.params.get("metallic").and_then(Value::as_f64).unwrap_or(0.0) as f32,
                ..Default::default()
            };
            history.push(scene);
            let id = scene.create_custom_mesh(name, mesh, Some(material));
            Ok(json!({
                "id": id,
                "name": scene.node(id).map(|n| n.name.clone()).unwrap_or_default(),
                "vertices": vertex_count,
                "triangles": triangle_count,
                "bounds": scene.node(id).and_then(|n| n.world_bounds().map(|(_, _, size)| size)),
            }))
        }

        "get_mesh_topology" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Mesh not found");
                return;
            };
            let limit = command.params.get("limit").and_then(Value::as_u64).unwrap_or(100) as usize;
            let Some(mesh) = scene.mesh(id) else {
                return_fail(command, "Node is not a mesh");
                return;
            };
            let mut edges = HashSet::new();
            for triangle in mesh.indices.chunks_exact(3) {
                for pair in [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])] {
                    edges.insert(if pair.0 < pair.1 { pair } else { (pair.1, pair.0) });
                }
            }
            let mut triangles = Vec::new();
            for (face, chunk) in mesh.indices.chunks_exact(3).take(limit).enumerate() {
                let corners: Vec<[f32; 3]> = chunk
                    .iter()
                    .filter_map(|index| mesh.positions.get(*index as usize).copied())
                    .collect();
                triangles.push(json!({ "face": face, "indices": chunk, "positions": corners }));
            }
            Ok(json!({
                "id": id,
                "vertices": mesh.positions.len(),
                "triangles": mesh.indices.len() / 3,
                "edges": edges.len(),
                "faces": triangles,
            }))
        }

        "extrude_faces" | "inset_faces" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Mesh not found");
                return;
            };
            let Some(faces) = parse_faces_as_vec(command.params.get("faces")) else {
                return_fail(command, "'faces' must be an array of triangle indices");
                return;
            };
            let extruding = command.method == "extrude_faces";
            let key = if extruding { "distance" } else { "amount" };
            let Some(value) = command.params.get(key).and_then(Value::as_f64) else {
                return_fail(command, &format!("'{key}' must be a number"));
                return;
            };
            let value = value as f32;
            history.push(scene);
            let outcome = match scene.mesh_mut(id) {
                Some(mesh) if extruding => mesh_ops::extrude_faces(mesh, &faces, value),
                Some(mesh) => mesh_ops::inset_faces(mesh, &faces, value),
                None => Err("Node is not a mesh".into()),
            };
            match outcome {
                Ok(()) => {
                    scene.revision += 1;
                    Ok(json!({
                        "id": id,
                        "operation": command.method,
                        "vertices": scene.mesh(id).map(|m| m.positions.len()).unwrap_or(0),
                        "triangles": scene.mesh(id).map(|m| m.indices.len() / 3).unwrap_or(0),
                        "bounds": scene.node(id).and_then(|n| n.world_bounds().map(|(_, _, size)| size)),
                    }))
                }
                Err(message) => Err(message),
            }
        }

        "weld_vertices" | "flip_normals" | "bevel_mesh" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Mesh not found");
                return;
            };
            let value = command.params.get("epsilon").or_else(|| command.params.get("amount"))
                .and_then(Value::as_f64).unwrap_or(0.01) as f32;
            history.push(scene);
            let outcome = match scene.mesh_mut(id) {
                Some(mesh) if command.method == "weld_vertices" => mesh_ops::merge_by_distance(mesh, value),
                Some(mesh) if command.method == "flip_normals" => mesh_ops::flip_normals(mesh),
                Some(mesh) => mesh_ops::bevel(mesh, value),
                None => Err("Node is not a mesh".into()),
            };
            match outcome {
                Ok(()) => {
                    scene.revision += 1;
                    Ok(json!({
                        "id": id,
                        "operation": command.method,
                        "vertices": scene.mesh(id).map(|m| m.positions.len()).unwrap_or(0),
                        "triangles": scene.mesh(id).map(|m| m.indices.len() / 3).unwrap_or(0),
                        "bounds": scene.node(id).and_then(|n| n.world_bounds().map(|(_, _, size)| size)),
                    }))
                }
                Err(message) => Err(message),
            }
        }

        "get_scene_info" => {
            let fields: Vec<String> = command
                .params
                .get("fields")
                .and_then(Value::as_array)
                .map(|array| {
                    array
                        .iter()
                        .filter_map(|value| value.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_else(|| {
                    ["location", "size", "ground", "rotation", "scale"]
                        .iter()
                        .map(|value| value.to_string())
                        .collect()
                });
            let limit = command.params.get("limit").and_then(Value::as_u64).unwrap_or(50) as usize;
            let wants = |name: &str| fields.iter().any(|field| field == name);
            let mut objects = Vec::new();
            for node in scene.nodes.iter().take(limit) {
                let mut entry = json!({
                    "name": node.name,
                    "type": kind_label(&node.kind),
                });
                if wants("location") {
                    entry["location"] = json!(node.transform.position);
                }
                if wants("rotation") {
                    entry["rotation_deg"] = json!(node.transform.euler().map(f32::to_degrees));
                }
                if wants("scale") {
                    entry["scale"] = json!(node.transform.scale);
                }
                if wants("size") || wants("ground") {
                    match node.world_bounds() {
                        Some((_, _, size)) => {
                            if wants("size") {
                                entry["size"] = json!(size);
                            }
                            if wants("ground") {
                                entry["ground"] = json!(scene::ground_status(
                                    node.world_bounds().map(|(min, _, _)| min[1]).unwrap_or(0.0)
                                ));
                            }
                        }
                        None => {
                            if wants("size") {
                                entry["size"] = Value::Null;
                            }
                            if wants("ground") {
                                entry["ground"] = json!("no mesh");
                            }
                        }
                    }
                }
                if wants("material") {
                    entry["material"] = match &node.kind {
                        NodeKind::Mesh { material, .. } => material.summary_json(),
                        _ => Value::Null,
                    };
                }
                objects.push(entry);
            }
            Ok(json!({
                "node_count": scene.nodes.len(),
                "objects": objects,
            }))
        }

        "get_modeling_guide" => Ok(json!({ "guide": MODELING_GUIDE })),

        "set_dimensions" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found");
                return;
            };
            let Some(size) = parse_array3(command.params.get("size")) else {
                return_fail(command, "'size' must be [width, height, depth]");
                return;
            };
            let Some(node) = scene.node(id).cloned() else {
                return_fail(command, "Node not found");
                return;
            };
            let local_size = match &node.kind {
                NodeKind::Mesh { mesh, .. } => {
                    if mesh.positions.is_empty() {
                        return_fail(command, "Mesh has no vertices");
                        return;
                    }
                    let mut min = [f32::INFINITY; 3];
                    let mut max = [f32::NEG_INFINITY; 3];
                    for point in &mesh.positions {
                        for axis in 0..3 {
                            min[axis] = min[axis].min(point[axis]);
                            max[axis] = max[axis].max(point[axis]);
                        }
                    }
                    [max[0] - min[0], max[1] - min[1], max[2] - min[2]]
                }
                _ => {
                    return_fail(command, "Node is not a mesh");
                    return;
                }
            };
            let new_scale = [
                if local_size[0].abs() > 1e-6 { size[0] / local_size[0] } else { 1.0 },
                if local_size[1].abs() > 1e-6 { size[1] / local_size[1] } else { 1.0 },
                if local_size[2].abs() > 1e-6 { size[2] / local_size[2] } else { 1.0 },
            ];
            history.push(scene);
            scene.transform_object(id, None, None, Some(new_scale));
            let bounds = scene.node(id).map(|node| node.bounds_json()).unwrap_or(Value::Null);
            Ok(json!({
                "id": id,
                "size": size,
                "scale": new_scale,
                "bounds": bounds,
            }))
        }

        "get_object_info" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found");
                return;
            };
            match scene.node(id) {
                Some(node) => Ok(node.to_json()),
                None => Err("Node not found".into()),
            }
        }

        "delete_object" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found");
                return;
            };
            history.push(scene);
            if scene.delete_node(id) {
                Ok(json!({ "id": id, "deleted": true }))
            } else {
                Err("Node not found".into())
            }
        }

        "rename_object" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found");
                return;
            };
            let new_name = command
                .params
                .get("new_name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if new_name.is_empty() {
                return_fail(command, "'new_name' must be a non-empty string");
                return;
            }
            history.push(scene);
            if scene.set_node_name(id, new_name.clone()) {
                Ok(json!({ "id": id, "name": new_name }))
            } else {
                Err("Node not found".into())
            }
        }

        "set_object_visibility" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found");
                return;
            };
            let Some(visible) = command.params.get("visible").and_then(Value::as_bool) else {
                return_fail(command, "'visible' must be a boolean");
                return;
            };
            history.push(scene);
            if scene.set_visible(id, visible) {
                Ok(json!({ "id": id, "visible": visible }))
            } else {
                Err("Node not found".into())
            }
        }

        "select_object" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found");
                return;
            };
            if scene.node(id).is_some() {
                scene.selected = Some(id);
                Ok(json!({
                    "id": id,
                    "name": scene.node(id).map(|n| n.name.clone()).unwrap_or_default(),
                }))
            } else {
                Err("Node not found".into())
            }
        }

        "apply_modifier" => {
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
                "mirror" => {
                    let axis = command.params.get("axis").and_then(Value::as_u64).unwrap_or(0) as usize;
                    match scene.mesh_mut(id) {
                        Some(mesh) => mesh_ops::mirror(mesh, axis),
                        None => Err("Node is not a mesh".into()),
                    }
                }
                "array" => {
                    let count = command.params.get("count").and_then(Value::as_u64).unwrap_or(3) as u32;
                    let offset = parse_array3(command.params.get("offset")).unwrap_or([2.0, 0.0, 0.0]);
                    let Some(source) = scene.mesh(id).cloned() else {
                        return_fail(command, "Node is not a mesh");
                        return;
                    };
                    let combined = mesh_ops::array(&source, count, offset);
                    match scene.mesh_mut(id) {
                        Some(mesh) => {
                            *mesh = combined;
                            Ok(())
                        }
                        None => Err("Node is not a mesh".into()),
                    }
                }
                "solidify" => {
                    let thickness = command.params.get("thickness").and_then(Value::as_f64).unwrap_or(0.1) as f32;
                    match scene.mesh_mut(id) {
                        Some(mesh) => mesh_ops::solidify(mesh, thickness),
                        None => Err("Node is not a mesh".into()),
                    }
                }
                "smooth" => {
                    let factor = command.params.get("factor").and_then(Value::as_f64).unwrap_or(0.5) as f32;
                    let iterations = command.params.get("iterations").and_then(Value::as_u64).unwrap_or(1) as u32;
                    match scene.mesh_mut(id) {
                        Some(mesh) => mesh_ops::smooth(mesh, factor, iterations),
                        None => Err("Node is not a mesh".into()),
                    }
                }
                "triangulate" => match scene.mesh_mut(id) {
                    Some(mesh) => mesh_ops::triangulate(mesh),
                    None => Err("Node is not a mesh".into()),
                },
                "subdivide" => {
                    let levels = command.params.get("levels").and_then(Value::as_u64).unwrap_or(1).min(3) as u32;
                    match scene.mesh_mut(id) {
                        Some(mesh) => mesh_ops::subdivide(mesh, levels),
                        None => Err("Node is not a mesh".into()),
                    }
                }
                "extrude" => {
                    let direction = parse_array3(command.params.get("direction")).unwrap_or([0.0, 1.0, 0.0]);
                    let distance = command.params.get("distance").and_then(Value::as_f64).unwrap_or(1.0) as f32;
                    match scene.mesh_mut(id) {
                        Some(mesh) => mesh_ops::extrude(mesh, direction, distance),
                        None => Err("Node is not a mesh".into()),
                    }
                }
                "recalculate_normals" => match scene.mesh_mut(id) {
                    Some(mesh) => {
                        mesh_ops::recalculate_normals(mesh);
                        Ok(())
                    }
                    None => Err("Node is not a mesh".into()),
                },
                other => Err(format!("Unknown modifier: {other}")),
            };
            match outcome {
                Ok(()) => {
                    scene.revision += 1;
                    Ok(json!({
                        "id": id,
                        "operation": op,
                        "vertices": scene.mesh(id).map(|m| m.positions.len()).unwrap_or(0),
                        "triangles": scene.mesh(id).map(|m| m.indices.len() / 3).unwrap_or(0),
                        "bounds": scene.node(id).and_then(|n| n.world_bounds().map(|(_, _, size)| size)),
                    }))
                }
                Err(message) => Err(message),
            }
        }

        "set_material_preset" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Node not found");
                return;
            };
            let preset = command
                .params
                .get("preset")
                .and_then(Value::as_str)
                .unwrap_or("plastic")
                .to_string();
            let (color, roughness, metallic) = match preset.as_str() {
                "plastic" => ([0.80, 0.80, 0.85, 1.0], 0.35, 0.0),
                "matte" => ([0.70, 0.70, 0.70, 1.0], 0.90, 0.0),
                "metal" => ([0.60, 0.60, 0.65, 1.0], 0.30, 0.85),
                "chrome" => ([0.90, 0.90, 0.95, 1.0], 0.05, 1.0),
                "glass" => ([0.60, 0.80, 0.90, 0.35], 0.05, 0.0),
                "rubber" => ([0.08, 0.08, 0.09, 1.0], 0.95, 0.0),
                "gold" => ([0.85, 0.65, 0.20, 1.0], 0.20, 1.0),
                "ceramic" => ([0.95, 0.95, 0.92, 1.0], 0.15, 0.0),
                "emissive" => ([1.0, 0.60, 0.20, 1.0], 0.40, 0.0),
                other => {
                    return_fail(command, &format!("Unknown preset: {other}"));
                    return;
                }
            };
            let color = parse_array4(command.params.get("color")).unwrap_or(color);
            history.push(scene);
            let applied = {
                if let Some(material) = scene.material_mut(id) {
                    material.base_color = color;
                    material.roughness = roughness;
                    material.metallic = metallic;
                    Some(material.summary_json())
                } else {
                    None
                }
            };
            match applied {
                Some(summary) => {
                    scene.revision += 1;
                    Ok(json!({ "id": id, "preset": preset, "material": summary }))
                }
                None => Err("Node is not a mesh".into()),
            }
        }

        "create_light" => {
            let kind = command
                .params
                .get("light_type")
                .and_then(Value::as_str)
                .unwrap_or("directional")
                .to_string();
            let light_type = match kind.as_str() {
                "directional" => LightType::Directional,
                "point" => LightType::Point,
                "ambient" => LightType::Ambient,
                other => {
                    return_fail(command, &format!("Unknown light type: {other}"));
                    return;
                }
            };
            let name = command.params.get("name").and_then(Value::as_str).map(str::to_string);
            let position = parse_array3(command.params.get("position"));
            let color3 = parse_array3(command.params.get("color"));
            let intensity = command.params.get("intensity").and_then(Value::as_f64).unwrap_or(1.0) as f32;
            history.push(scene);
            let id = scene.spawn_light(light_type, name);
            if let Some(p) = position {
                let _ = scene.transform_object(id, Some(p), None, None);
            }
            if let Some(node) = scene.node_mut(id) {
                if let NodeKind::Light { color, intensity: slot, .. } = &mut node.kind {
                    if let Some(c) = color3 {
                        *color = c;
                    }
                    *slot = intensity;
                }
            }
            scene.revision += 1;
            Ok(json!({
                "id": id,
                "name": scene.node(id).map(|n| n.name.clone()).unwrap_or_default(),
                "light_type": kind,
            }))
        }

        "set_light" => {
            let Some(id) = resolve_id(scene, &command.params) else {
                return_fail(command, "Light not found");
                return;
            };
            let is_light = scene.node(id).is_some_and(|n| matches!(n.kind, NodeKind::Light { .. }));
            if !is_light {
                return_fail(command, "Node is not a light");
                return;
            }
            let color3 = parse_array3(command.params.get("color"));
            let intensity = command.params.get("intensity").and_then(Value::as_f64).map(|v| v as f32);
            let position = parse_array3(command.params.get("position"));
            history.push(scene);
            {
                if let Some(node) = scene.node_mut(id) {
                    if let NodeKind::Light { color, intensity: slot, .. } = &mut node.kind {
                        if let Some(c) = color3 {
                            *color = c;
                        }
                        if let Some(value) = intensity {
                            *slot = value;
                        }
                    }
                    if let Some(p) = position {
                        node.transform.position = p;
                    }
                }
            }
            scene.revision += 1;
            Ok(json!({ "id": id, "light": scene.node(id).map(|n| n.to_json()).unwrap_or_default() }))
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

const MODELING_GUIDE: &str = "\
FORGE MODELING GUIDE — Y is up, X/Z are the ground plane, 1 unit = 1 metre.\n\
BLENDER-STYLE TOPOLOGY: for anything more complex than a simple prop, build a real mesh with\n\
create_mesh_from_faces (explicit vertices + polygon faces) instead of stacking boxes and cylinders.\n\
That tool behaves like bmesh.from_pydata: faces are vertex-index lists and are triangulated for you.\n\
Then use get_mesh_topology to find triangle face indices, and edit with extrude_faces,\n\
inset_faces, weld_vertices, flip_normals and bevel_mesh. Stacked primitives look like Roblox;\n\
topology editing looks like Blender. Reserve spawn_primitive for wheels, bolts and simple props.\n\
PRIMITIVE LOCAL DIMENSIONS (before scale):\n\
- cube: 1 x 1 x 1, centred at origin (-0.5..0.5 on each axis)\n\
- sphere: diameter 1\n\
- cylinder: radius 0.5, height 1 along Y\n\
- cone: radius 0.5, height 1 along Y\n\
- torus: major radius 0.5, tube 0.18; lies flat in XZ with height 0.36\n\
- plane: 1 x 1 on XZ with ZERO thickness — use a thin cube or the solidify modifier for solids\n\
POSITION sets the object's CENTER. SCALE MULTIPLIES local dimensions: a desired world size\n\
[w, h, d] means scale [w, h, d] for cube/sphere/cylinder/cone, and [w/1.36, h/0.36, d/1.36]\n\
for torus. Prefer set_dimensions({size:[w,h,d]}) so the engine computes scale for you.\n\
GROUND CONTACT: 'ground' in bounds is derived from the AABB's lowest point (min.y). To rest an\n\
object on the ground, set its centre Y to half its height (a 4x0.6x1 box -> centre y=0.3).\n\
SELF-CHECK PROCEDURE: 1) build with build_model or spawn_primitive; 2) call get_scene_info and\n\
check every part's size and ground status; 3) call render_viewport_snapshot with views\n\
'front' (from +Z), 'right' (from +X), 'top' and 'three_quarter' to see it from several angles;\n\
4) fix anything\n\
floating, below ground, or zero-thickness; 5) finish with get_scene_hierarchy.\n\
NEVER stack zero-thickness planes to fake a solid, leave parts floating, or guess scale\n\
without reading back bounds. Use get_object_info to inspect one part in detail.\n";

fn kind_label(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::Empty => "Empty",
        NodeKind::Mesh { .. } => "Mesh",
        NodeKind::Camera => "Camera",
        NodeKind::Light { .. } => "Light",
    }
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

fn parse_faces(value: Option<&Value>) -> Option<Vec<Vec<usize>>> {
    value?
        .as_array()?
        .iter()
        .map(|face| {
            face.as_array()?
                .iter()
                .map(|index| index.as_u64().map(|value| value as usize))
                .collect()
        })
        .collect()
}

fn parse_faces_as_vec(value: Option<&Value>) -> Option<Vec<usize>> {
    value?
        .as_array()?
        .iter()
        .map(|index| index.as_u64().map(|value| value as usize))
        .collect()
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

