//! three-d rendering layer. Everything here targets the OpenGL 3.3 Core
//! context that three-d creates by default (its shaders are `#version 330 core`).

use crate::scene::{NodeKind, Scene, TextureData as SceneTexture};
use crate::ui::ShadingMode;
use three_d::*;

pub struct Renderer {
    context: Context,
    built_revision: u64,
    built_shading: ShadingMode,
    lit: Vec<Gm<Mesh, PhysicalMaterial>>,
    unlit: Vec<Gm<Mesh, ColorMaterial>>,
    wires: Vec<Wireframe>,
    selection: Vec<Wireframe>,
    grid: Gm<Mesh, ColorMaterial>,
    ambient: AmbientLight,
    directional: DirectionalLight,
}

impl Renderer {
    pub fn new(context: &Context, scene: &Scene) -> Self {
        let grid = build_grid(context);
        let ambient = AmbientLight::new(context, 0.45, Srgba::WHITE);
        let directional = DirectionalLight::new(
            context,
            1.2,
            Srgba::new_opaque(255, 246, 230),
            vec3(-0.5, -1.0, -0.4),
        );
        let mut renderer = Self {
            context: context.clone(),
            built_revision: u64::MAX,
            built_shading: ShadingMode::Lit,
            lit: Vec::new(),
            unlit: Vec::new(),
            wires: Vec::new(),
            selection: Vec::new(),
            grid,
            ambient,
            directional,
        };
        renderer.rebuild(scene, ShadingMode::Lit);
        renderer
    }

    /// Rebuild GPU-side objects when the scene or shading mode changed.
    pub fn sync(&mut self, scene: &Scene, shading: ShadingMode) {
        if self.built_revision != scene.revision || self.built_shading != shading {
            self.rebuild(scene, shading);
        }
    }

    fn rebuild(&mut self, scene: &Scene, shading: ShadingMode) {
        self.lit.clear();
        self.unlit.clear();
        self.wires.clear();
        self.selection.clear();

        for node in &scene.nodes {
            let (mesh, material) = match &node.kind {
                NodeKind::Mesh { mesh, material, .. } if node.visible => (mesh, material),
                _ => continue,
            };
            let cpu = cpu_mesh_from(mesh, &node.transform.matrix());
            let srgba = srgba(&material.base_color);

            match shading {
                ShadingMode::Lit => {
                    let mut physical = PhysicalMaterial::new_opaque(
                        &self.context,
                        &CpuMaterial {
                            albedo: srgba,
                            roughness: material.roughness,
                            metallic: material.metallic,
                            ..Default::default()
                        },
                    );
                    if let Some(tex) = &material.albedo {
                        physical.albedo_texture = Some(texture_ref(&self.context, tex, true));
                    }
                    if let Some(tex) = &material.roughness_map {
                        physical.metallic_roughness_texture =
                            Some(texture_ref(&self.context, tex, false));
                    }
                    if let Some(tex) = &material.normal_map {
                        physical.normal_texture = Some(texture_ref(&self.context, tex, false));
                    }
                    self.lit.push(Gm::new(Mesh::new(&self.context, &cpu), physical));
                }
                ShadingMode::Unlit => {
                    let unlit = ColorMaterial {
                        color: srgba,
                        texture: material
                            .albedo
                            .as_ref()
                            .map(|tex| texture_ref(&self.context, tex, true)),
                        ..Default::default()
                    };
                    self.unlit.push(Gm::new(Mesh::new(&self.context, &cpu), unlit));
                }
                ShadingMode::Wireframe => {
                    self.wires.push(Wireframe::new_from_cpu_mesh(&self.context, &cpu, 1.0, srgba));
                }
            }

            if scene.selected == Some(node.id) {
                let highlight = Srgba::new_opaque(255, 180, 40);
                self.selection
                    .push(Wireframe::new_from_cpu_mesh(&self.context, &cpu, 1.0, highlight));
            }
        }

        // Reflect the first scene light in the directional light direction.
        for node in &scene.nodes {
            if let NodeKind::Light { light_type, intensity, color } = &node.kind {
                if !node.visible {
                    continue;
                }
                let dir = vec3(
                    -node.transform.position[0],
                    -node.transform.position[1].max(0.1),
                    -node.transform.position[2],
                );
                self.directional = DirectionalLight::new(
                    &self.context,
                    *intensity,
                    Srgba::new_opaque(
                        (color[0].clamp(0.0, 1.0) * 255.0) as u8,
                        (color[1].clamp(0.0, 1.0) * 255.0) as u8,
                        (color[2].clamp(0.0, 1.0) * 255.0) as u8,
                    ),
                    dir.normalize(),
                );
                if *light_type == crate::scene::LightType::Ambient {
                    self.ambient = AmbientLight::new(&self.context, *intensity, Srgba::WHITE);
                }
                break;
            }
        }

        self.built_revision = scene.revision;
        self.built_shading = shading;
    }

    pub fn render(
        &self,
        frame_input: &FrameInput,
        scene: &Scene,
        camera: &mut Camera,
        shading: ShadingMode,
        show_grid: bool,
        gui: &GUI,
    ) {
        let bg = scene.background;
        let screen = frame_input.screen();
        let mut target = screen.clear(ClearState::color_and_depth(
            bg[0], bg[1], bg[2], 1.0, 1.0,
        ));

        match shading {
            ShadingMode::Lit => {
                if !self.lit.is_empty() {
                    target = target.render(
                        &mut *camera,
                        &self.lit,
                        &[&self.ambient, &self.directional],
                    );
                }
            }
            ShadingMode::Unlit => {
                if !self.unlit.is_empty() {
                    target = target.render(&mut *camera, &self.unlit, &[]);
                }
            }
            ShadingMode::Wireframe => {
                if !self.wires.is_empty() {
                    target = target.render(&mut *camera, &self.wires, &[]);
                }
            }
        }

        if !self.selection.is_empty() {
            target = target.render(&mut *camera, &self.selection, &[]);
        }
        if show_grid {
            target = target.render(&mut *camera, &[&self.grid], &[]);
        }

        target.write(|| gui.render()).unwrap();
    }
}

fn srgba(color: &[f32; 4]) -> Srgba {
    Srgba::new(
        (color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (color[2].clamp(0.0, 1.0) * 255.0) as u8,
        (color[3].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

/// Turn a CPU RGBA8 texture into a GPU texture reference for a material.
/// Colour/albedo textures are converted to linear sRGB; data maps are not.
fn texture_ref(context: &Context, tex: &SceneTexture, srgb: bool) -> Texture2DRef {
    let mut cpu = CpuTexture {
        name: tex.name.clone(),
        data: TextureData::RgbaU8(tex.pixels.clone()),
        width: tex.width,
        height: tex.height,
        ..Default::default()
    };
    if srgb {
        cpu.data.to_linear_srgb();
    }
    Texture2DRef::from_cpu_texture(context, &cpu)
}

fn cpu_mesh_from(data: &crate::scene::MeshData, transform: &glam::Mat4) -> CpuMesh {
    // three-d uses cgmath; rebuild the column-major matrix from glam's array.
    let a = transform.to_cols_array();
    let cg = Mat4::from_cols(
        vec4(a[0], a[1], a[2], a[3]),
        vec4(a[4], a[5], a[6], a[7]),
        vec4(a[8], a[9], a[10], a[11]),
        vec4(a[12], a[13], a[14], a[15]),
    );
    let mut cpu = CpuMesh {
        positions: Positions::F32(
            data.positions
                .iter()
                .map(|p| vec3(p[0], p[1], p[2]))
                .collect(),
        ),
        normals: Some(
            data.normals.iter().map(|n| vec3(n[0], n[1], n[2])).collect(),
        ),
        uvs: Some(data.uvs.iter().map(|u| vec2(u[0], u[1])).collect()),
        indices: Indices::U32(data.indices.clone()),
        ..Default::default()
    };
    let _ = cpu.transform(cg);
    cpu
}

/// Thin-quad grid on the XZ plane plus coloured axis lines.
fn build_grid(context: &Context) -> Gm<Mesh, ColorMaterial> {
    const EXTENT: f32 = 20.0f32;
    const STEP: i32 = 20;
    const THICK: f32 = 0.012;

    let mut positions: Vec<Vector3<f32>> = Vec::new();
    let mut colors: Vec<Srgba> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let mut quad = |a: Vector3<f32>, b: Vector3<f32>, color: Srgba| {
        let base = positions.len() as u32;
        let dir = b - a;
        // Extrude the segment slightly in Y and X/Z to form a thin ribbon.
        let up = vec3(0.0, THICK, 0.0);
        positions.extend_from_slice(&[a, a + up, b + up, b]);
        colors.extend_from_slice(&[color; 4]);
        let _ = dir;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    for i in -STEP..=STEP {
        let p = i as f32;
        let color = if i == 0 {
            Srgba::new_opaque(90, 90, 110)
        } else {
            Srgba::new_opaque(55, 55, 62)
        };
        quad(vec3(-EXTENT, 0.0, p), vec3(EXTENT, 0.0, p), color);
        let color = if i == 0 {
            Srgba::new_opaque(90, 90, 110)
        } else {
            Srgba::new_opaque(55, 55, 62)
        };
        quad(vec3(p, 0.0, -EXTENT), vec3(p, 0.0, EXTENT), color);
    }

    // Axis markers.
    quad(vec3(-EXTENT, 0.002, 0.0), vec3(EXTENT, 0.002, 0.0), Srgba::new_opaque(170, 60, 60));
    quad(vec3(0.0, 0.002, -EXTENT), vec3(0.0, 0.002, EXTENT), Srgba::new_opaque(60, 100, 180));

    let cpu = CpuMesh {
        positions: Positions::F32(positions),
        colors: Some(colors),
        indices: Indices::U32(indices),
        ..Default::default()
    };
    Gm::new(
        Mesh::new(context, &cpu),
        ColorMaterial { color: Srgba::WHITE, ..Default::default() },
    )
}
