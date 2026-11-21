//! Minimalist Blender-style editor shell built with the egui version that
//! three-d re-exports (`three_d::egui`).

use crate::scene::{History, Material, NodeKind, OrbitCamera, PrimitiveKind, Scene, Transform};
use three_d::egui::{self, Button, Color32, DragValue, Panel, Ui};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AppMode {
    Object,
    Edit,
    Pose,
}

impl AppMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Object => "Object Mode",
            Self::Edit => "Edit Mode",
            Self::Pose => "Pose Mode",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShadingMode {
    Lit,
    Unlit,
    Wireframe,
}

impl ShadingMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Lit => "Lit",
            Self::Unlit => "Unlit",
            Self::Wireframe => "Wireframe",
        }
    }
}

#[derive(Debug)]
pub struct UiState {
    pub mode: AppMode,
    pub shading: ShadingMode,
    pub show_grid: bool,
    pub status: String,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            mode: AppMode::Object,
            shading: ShadingMode::Lit,
            show_grid: true,
            status: "Ready".into(),
        }
    }
}

/// Draw the complete editor UI into three-d's full-screen root Ui.
pub fn draw_ui(
    root_ui: &mut Ui,
    scene: &mut Scene,
    history: &mut History,
    _camera: &mut OrbitCamera,
    ui_state: &mut UiState,
    mcp_connected: bool,
    gpu_summary: &str,
) {
    let ctx = root_ui.ctx();
    handle_shortcuts(ctx, scene, history, ui_state);

    top_bar(root_ui, scene, history, ui_state, mcp_connected, gpu_summary);

    // One undo snapshot per frame for any mutation coming from the panels.
    // Captured after the top bar because Undo/Redo buttons mutate history there.
    let before = scene.clone();
    let revision_before = scene.revision;

    outliner(root_ui, scene);
    pose_panel(root_ui, scene, ui_state.mode);
    inspector(root_ui, scene);
    timeline(root_ui, scene);

    if scene.revision != revision_before {
        history.push(&before);
    }
}

fn handle_shortcuts(ctx: &egui::Context, scene: &mut Scene, history: &mut History, state: &mut UiState) {
    let action = ctx.input(|input| {
        let ctrl = input.modifiers.ctrl || input.modifiers.command;
        if ctrl && input.key_pressed(egui::Key::Z) {
            Some(if input.modifiers.shift { "redo" } else { "undo" })
        } else if ctrl && input.key_pressed(egui::Key::Y) {
            Some("redo")
        } else {
            None
        }
    });

    match action {
        Some("undo") => {
            if history.undo(scene) {
                scene.revision += 1;
                state.status = "Undo".into();
            }
        }
        Some("redo") => {
            if history.redo(scene) {
                scene.revision += 1;
                state.status = "Redo".into();
            }
        }
        _ => {}
    }
}

fn top_bar(
    root_ui: &mut Ui,
    scene: &mut Scene,
    history: &mut History,
    state: &mut UiState,
    mcp_connected: bool,
    gpu_summary: &str,
) {
    Panel::top("top_bar").show_inside(root_ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("FORGE").strong().color(Color32::from_rgb(240, 170, 60)));
            ui.separator();

            for mode in [AppMode::Object, AppMode::Edit, AppMode::Pose] {
                if ui.selectable_label(state.mode == mode, mode.label()).clicked() {
                    state.mode = mode;
                    state.status = format!("Mode: {}", mode.label());
                }
            }
            ui.separator();

            ui.label("Shading:");
            for shading in [ShadingMode::Lit, ShadingMode::Unlit, ShadingMode::Wireframe] {
                if ui.selectable_label(state.shading == shading, shading.label()).clicked() {
                    state.shading = shading;
                    state.status = format!("Shading: {}", shading.label());
                }
            }
            ui.separator();

            if ui.button("Undo").on_hover_text("Ctrl+Z").clicked() {
                if history.undo(scene) {
                    scene.revision += 1;
                    state.status = "Undo".into();
                }
            }
            if ui.button("Redo").on_hover_text("Ctrl+Y / Ctrl+Shift+Z").clicked() {
                if history.redo(scene) {
                    scene.revision += 1;
                    state.status = "Redo".into();
                }
            }
            ui.separator();

            ui.add_enabled(false, Button::new("Import glTF"))
                .on_hover_text("Available in Phase 4");
            ui.add_enabled(false, Button::new("Export glTF"))
                .on_hover_text("Available in Phase 4");
            ui.checkbox(&mut state.show_grid, "Grid");

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (dot, label) = if mcp_connected {
                    (Color32::from_rgb(80, 220, 120), "MCP connected")
                } else {
                    (Color32::from_rgb(220, 90, 90), "MCP idle")
                };
                ui.label(egui::RichText::new(label).color(dot).small());
                ui.label(egui::RichText::new(gpu_summary).small().weak());
                ui.label(egui::RichText::new(&state.status).small().weak());
            });
        });
    });
}

fn outliner(root_ui: &mut Ui, scene: &mut Scene) {
    Panel::left("outliner_panel")
        .default_size(230.0)
        .min_size(150.0)
        .show_inside(root_ui, |ui| {
            ui.heading("Outliner");
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                let ids: Vec<u32> = scene.nodes.iter().map(|n| n.id).collect();
                for id in ids {
                    let (name, kind, visible, selected) = match scene.node(id) {
                        Some(n) => (
                            n.name.clone(),
                            icon_for(&n.kind).to_string(),
                            n.visible,
                            scene.selected == Some(id),
                        ),
                        None => continue,
                    };
                    ui.horizontal(|ui| {
                        let mut vis = visible;
                        if ui.checkbox(&mut vis, "").changed() {
                            let _ = scene.set_visible(id, vis);
                        }
                        if ui.selectable_label(selected, format!("{kind} {name}")).clicked() {
                            scene.selected = Some(id);
                        }
                    });
                }
            });
        });
}

fn pose_panel(root_ui: &mut Ui, scene: &mut Scene, mode: AppMode) {
    if mode != AppMode::Pose {
        return;
    }
    Panel::left("pose_panel")
        .default_size(210.0)
        .show_inside(root_ui, |ui| {
            ui.heading("Pose Mode");
            ui.separator();
            let id = scene.selected;
            let mut names = Vec::new();
            let mut current = Vec::new();
            if let Some(id) = id {
                if let Some(node) = scene.node(id) {
                    if let NodeKind::Armature { armature, pose } = &node.kind {
                        names = armature.bones.iter().map(|bone| bone.name.clone()).collect::<Vec<_>>();
                        current = pose.clone();
                    }
                }
            }
            if names.is_empty() {
                ui.label(egui::RichText::new("Select an armature").weak().italics());
                return;
            }
            let Some(id) = id else { return; };
            let mut edits: Vec<(usize, Transform)> = Vec::new();
            for (index, name) in names.iter().enumerate() {
                ui.label(egui::RichText::new(name).small());
                let mut transform = current.get(index).copied().unwrap_or_default();
                let mut euler = transform.euler();
                ui.horizontal(|ui| {
                    for (axis_index, axis) in ["X", "Y", "Z"].iter().enumerate() {
                        ui.label(*axis);
                        ui.add(DragValue::new(&mut euler[axis_index]).speed(0.01));
                    }
                });
                if euler != transform.euler() {
                    transform.set_euler(euler[0], euler[1], euler[2]);
                    edits.push((index, transform));
                }
            }
            for (index, transform) in edits {
                let _ = scene.set_bone_pose(id, index, transform);
            }
        });
}

fn icon_for(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::Empty => "◇",
        NodeKind::Mesh { .. } => "▲",
        NodeKind::Camera => "▣",
        NodeKind::Light { .. } => "☀",
        NodeKind::Armature { .. } => "✳",
    }
}

fn inspector(root_ui: &mut Ui, scene: &mut Scene) {
    Panel::right("inspector_panel")
        .default_size(280.0)
        .min_size(200.0)
        .show_inside(root_ui, |ui| {
            ui.heading("Properties");
            ui.separator();

            let Some(id) = scene.selected else {
                ui.label(egui::RichText::new("Select an object in the Outliner").weak().italics());
                return;
            };
            let Some(node) = scene.node(id) else {
                scene.selected = None;
                return;
            };

            // Clone everything we need so `scene` can be mutated inside closures.
            let mut name = node.name.clone();
            let kind_label = kind_name(&node.kind).to_string();
            let mut transform = node.transform;
            let mesh_info: Option<(Material, Option<PrimitiveKind>)> = match &node.kind {
                NodeKind::Mesh { material, primitive, .. } => Some((material.clone(), *primitive)),
                _ => None,
            };
            let light_info = match &node.kind {
                NodeKind::Light { intensity, color, .. } => Some((*intensity, *color)),
                _ => None,
            };

            ui.horizontal(|ui| {
                ui.label("Name");
                if ui.text_edit_singleline(&mut name).changed() {
                    let _ = scene.set_node_name(id, name.clone());
                }
            });
            ui.label(egui::RichText::new(format!("Type: {kind_label}")).small().weak());
            ui.separator();

            ui.collapsing("Transform", |ui| {
                transform_widgets(ui, &mut transform);
                if scene.node(id).is_some_and(|n| n.transform != transform) {
                    if let Some(n) = scene.node_mut(id) {
                        n.transform = transform;
                    }
                    scene.revision += 1;
                }
            });

            if let Some((material, primitive)) = mesh_info {
                ui.collapsing("Material", |ui| {
                    let mut color = material.base_color;
                    let mut roughness = material.roughness;
                    let mut metallic = material.metallic;

                    ui.horizontal(|ui| {
                        ui.label("Base color");
                        for (i, label) in ["R", "G", "B", "A"].iter().enumerate() {
                            ui.label(*label);
                            ui.add(
                                DragValue::new(&mut color[i])
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .fixed_decimals(2),
                            );
                        }
                    });
                    let preview = Color32::from_rgba_premultiplied(
                        (color[0].clamp(0.0, 1.0) * 255.0) as u8,
                        (color[1].clamp(0.0, 1.0) * 255.0) as u8,
                        (color[2].clamp(0.0, 1.0) * 255.0) as u8,
                        (color[3].clamp(0.0, 1.0) * 255.0) as u8,
                    );
                    ui.colored_label(preview, "■ RGBA preview");

                    ui.horizontal(|ui| {
                        ui.label("Roughness");
                        ui.add(DragValue::new(&mut roughness).speed(0.01).range(0.0..=1.0));
                        ui.label("Metallic");
                        ui.add(DragValue::new(&mut metallic).speed(0.01).range(0.0..=1.0));
                    });

                    if let Some(kind) = primitive {
                        ui.label(
                            egui::RichText::new(format!("Primitive: {}", kind.as_str()))
                                .small()
                                .weak(),
                        );
                    }
                    for (label, texture) in [
                        ("Albedo", &material.albedo),
                        ("Roughness", &material.roughness_map),
                        ("Normal", &material.normal_map),
                    ] {
                        match texture {
                            Some(tex) => ui.label(
                                egui::RichText::new(format!(
                                    "{label}: {} ({}x{})",
                                    tex.name, tex.width, tex.height
                                ))
                                .small(),
                            ),
                            None => ui.label(
                                egui::RichText::new(format!("{label}: none")).small().weak(),
                            ),
                        };
                    }

                    if color != material.base_color
                        || (roughness - material.roughness).abs() > f32::EPSILON
                        || (metallic - material.metallic).abs() > f32::EPSILON
                    {
                        {
                            if let Some(slot) = scene.material_mut(id) {
                                slot.base_color = color;
                                slot.roughness = roughness;
                                slot.metallic = metallic;
                            }
                        }
                        scene.revision += 1;
                    }
                });
            }

            if let Some((intensity, color)) = light_info {
                ui.collapsing("Light", |ui| {
                    let mut i = intensity;
                    ui.horizontal(|ui| {
                        ui.label("Intensity");
                        ui.add(DragValue::new(&mut i).speed(0.05));
                    });
                    if (i - intensity).abs() > f32::EPSILON {
                        if let Some(NodeKind::Light { intensity: slot, .. }) =
                            scene.node_mut(id).map(|n| &mut n.kind)
                        {
                            *slot = i;
                        }
                        scene.revision += 1;
                    }
                    ui.label(format!("Color: {:?}", color));
                });
            }

            ui.separator();
            if ui.button("Delete object").clicked() {
                let _ = scene.delete_node(id);
            }
        });
}

fn kind_name(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::Empty => "Empty",
        NodeKind::Mesh { .. } => "Mesh",
        NodeKind::Camera => "Camera",
        NodeKind::Light { .. } => "Light",
        NodeKind::Armature { .. } => "Armature",
    }
}

fn transform_widgets(ui: &mut Ui, t: &mut Transform) {
    egui::Grid::new("transform_grid").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
        ui.label("Position");
        ui.horizontal(|ui| {
            for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*axis);
                ui.add(DragValue::new(&mut t.position[i]).speed(0.02));
            }
        });
        ui.end_row();

        let mut euler = t.euler();
        ui.label("Rotation (rad)");
        ui.horizontal(|ui| {
            for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*axis);
                ui.add(DragValue::new(&mut euler[i]).speed(0.01));
            }
        });
        if euler != t.euler() {
            t.set_euler(euler[0], euler[1], euler[2]);
        }
        ui.end_row();

        ui.label("Scale");
        ui.horizontal(|ui| {
            for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*axis);
                ui.add(DragValue::new(&mut t.scale[i]).speed(0.02));
            }
        });
        ui.end_row();
    });
}

fn timeline(root_ui: &mut Ui, scene: &mut Scene) {
    Panel::bottom("timeline_panel").default_size(120.0).min_size(70.0).show_inside(
        root_ui,
        |ui| {
            ui.horizontal(|ui| {
                let play_label = if scene.timeline.playing { "⏸ Pause" } else { "▶ Play" };
                if ui.button(play_label).clicked() {
                    scene.timeline.playing = !scene.timeline.playing;
                }
                ui.label("Frame");
                ui.add(
                    DragValue::new(&mut scene.timeline.frame)
                        .speed(1.0)
                        .range(scene.timeline.frame_start..=scene.timeline.frame_end),
                );
                ui.label("Start");
                ui.add(DragValue::new(&mut scene.timeline.frame_start).speed(1.0));
                ui.label("End");
                ui.add(DragValue::new(&mut scene.timeline.frame_end).speed(1.0));
                ui.label("FPS");
                ui.add(DragValue::new(&mut scene.timeline.fps).speed(1.0).range(1..=120));
                ui.label(egui::RichText::new("Keyframes arrive in Phase 4").small().weak());
            });

            let rect = ui.available_rect_before_wrap();
            let painter = ui.painter();
            painter.rect_filled(rect, 4.0, Color32::from_rgb(30, 30, 34));
            let range = (scene.timeline.frame_end - scene.timeline.frame_start).max(1) as f32;
            for i in 0..=10 {
                let x = rect.left() + rect.width() * (i as f32 / 10.0);
                painter.line_segment(
                    [egui::pos2(x, rect.top() + 6.0), egui::pos2(x, rect.bottom() - 6.0)],
                    egui::Stroke::new(1.0_f32, Color32::from_gray(70)),
                );
            }
            let t = (scene.timeline.frame - scene.timeline.frame_start) as f32 / range;
            let x = rect.left() + rect.width() * t.clamp(0.0, 1.0);
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(2.0_f32, Color32::from_rgb(240, 170, 60)),
            );
        },
    );
}
