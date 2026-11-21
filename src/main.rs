//! Forge — a Blender-inspired editor with an embedded MCP server.
//!
//! The main thread owns the window, scene and renderer. A dedicated tokio
//! thread runs the stdio MCP server and forwards tool calls here over channels.

mod ipc;
mod mcp;
mod render;
mod scene;
mod ui;

use ipc::AppCommand;
use render::Renderer;
use scene::{History, NodeId, OrbitCamera, PrimitiveKind, Scene, Transform};
use serde_json::{json, Value};
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
    eprintln!("[forge] OpenGL context created");

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
            handle_command(command, &mut scene, &mut history);
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
                ui::draw_ui(egui_ui, &mut scene, &mut history, &mut camera, &mut ui_state, true);
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

fn handle_command(command: AppCommand, scene: &mut Scene, history: &mut History) {
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

