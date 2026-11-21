//! Scene graph, primitives and history.
//!
//! This module is deliberately free of any rendering API so it can be reasoned
//! about (and tested) independently of `three-d`.

use crate::rig::Armature;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

pub type NodeId = u32;

const TAU: f32 = std::f32::consts::TAU;

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Transform {
    /// x, y, z
    pub position: [f32; 3],
    /// x, y, z, w quaternion
    pub rotation: [f32; 4],
    /// x, y, z
    pub scale: [f32; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }
    }
}

impl Transform {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::from(self.scale),
            Quat::from_xyzw(self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3]),
            Vec3::from(self.position),
        )
    }

    /// Euler angles in radians (XYZ order) -> quaternion.
    pub fn set_euler(&mut self, x: f32, y: f32, z: f32) {
        self.rotation = Quat::from_rotation_x(x)
            .mul_quat(Quat::from_rotation_y(y))
            .mul_quat(Quat::from_rotation_z(z))
            .to_array();
    }

    pub fn euler(&self) -> [f32; 3] {
        Quat::from_xyzw(self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3])
            .to_euler(glam::EulerRot::XYZ)
            .into()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "position": self.position,
            "rotation_xyzw": self.rotation,
            "scale": self.scale,
            "euler_xyz": self.euler(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn empty() -> Self {
        Self { positions: Vec::new(), normals: Vec::new(), uvs: Vec::new(), indices: Vec::new() }
    }

    /// Recompute smooth vertex normals from the index buffer.
    #[allow(dead_code)] // exposed through `modify_mesh` in Phase 2
    pub fn recalculate_normals(&mut self) {
        let mut sums = vec![[0.0f32; 3]; self.positions.len()];
        for tri in self.indices.chunks_exact(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if a >= sums.len() || b >= sums.len() || c >= sums.len() {
                continue;
            }
            let pa = Vec3::from(self.positions[a]);
            let pb = Vec3::from(self.positions[b]);
            let pc = Vec3::from(self.positions[c]);
            let n = (pb - pa).cross(pc - pa);
            for idx in [a, b, c] {
                sums[idx][0] += n.x;
                sums[idx][1] += n.y;
                sums[idx][2] += n.z;
            }
        }
        self.normals = sums
            .into_iter()
            .map(|n| {
                let v = Vec3::from(n);
                if v.length_squared() > 1e-12 {
                    let v = v.normalize();
                    [v.x, v.y, v.z]
                } else {
                    [0.0, 1.0, 0.0]
                }
            })
            .collect();
    }
}

/// CPU-side RGBA8 texture. Always downscaled to <= 1024x1024 before GPU upload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextureData {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA8 pixels, top-left origin.
    pub pixels: Vec<[u8; 4]>,
}

impl TextureData {
    pub fn summary_json(&self) -> Value {
        json!({
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "channels": 4,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Material {
    /// Linear RGBA base colour multiplier.
    pub base_color: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub albedo: Option<TextureData>,
    pub roughness_map: Option<TextureData>,
    pub normal_map: Option<TextureData>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [0.72, 0.74, 0.78, 1.0],
            roughness: 0.75,
            metallic: 0.0,
            albedo: None,
            roughness_map: None,
            normal_map: None,
        }
    }
}

impl Material {
    pub fn summary_json(&self) -> Value {
        json!({
            "base_color": self.base_color,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "albedo": self.albedo.as_ref().map(TextureData::summary_json),
            "roughness_map": self.roughness_map.as_ref().map(TextureData::summary_json),
            "normal_map": self.normal_map.as_ref().map(TextureData::summary_json),
        })
    }
}

/// Deform weights binding a mesh to an armature (Blender-style vertex groups).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkinBinding {
    pub armature: NodeId,
    /// Up to four bone indices per vertex.
    pub joints: Vec<[u16; 4]>,
    /// Corresponding normalized weights per vertex.
    pub weights: Vec<[f32; 4]>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrimitiveKind {
    Cube,
    Sphere,
    Cylinder,
    Cone,
    Plane,
    Torus,
}

impl PrimitiveKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cube" | "box" => Some(Self::Cube),
            "sphere" | "uvsphere" => Some(Self::Sphere),
            "cylinder" => Some(Self::Cylinder),
            "cone" => Some(Self::Cone),
            "plane" => Some(Self::Plane),
            "torus" => Some(Self::Torus),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cube => "cube",
            Self::Sphere => "sphere",
            Self::Cylinder => "cylinder",
            Self::Cone => "cone",
            Self::Plane => "plane",
            Self::Torus => "torus",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LightType {
    Directional,
    Point,
    Ambient,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeKind {
    Empty,
    Mesh {
        mesh: MeshData,
        material: Material,
        primitive: Option<PrimitiveKind>,
    },
    Camera,
    Light {
        light_type: LightType,
        color: [f32; 3],
        intensity: f32,
    },
    Armature {
        armature: Armature,
        /// One local transform per bone; defaults to the rest pose.
        pose: Vec<Transform>,
    },
}

impl NodeKind {
    fn label(&self) -> &'static str {
        match self {
            Self::Empty => "Empty",
            Self::Mesh { .. } => "Mesh",
            Self::Camera => "Camera",
            Self::Light { .. } => "Light",
            Self::Armature { .. } => "Armature",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    pub parent: Option<NodeId>,
    pub transform: Transform,
    pub visible: bool,
    pub kind: NodeKind,
}

/// Human-readable placement status derived from the lowest point of a mesh.
pub fn ground_status(min_y: f32) -> String {
    if min_y.abs() <= 0.01 {
        "on ground".to_string()
    } else if min_y > 0.0 {
        format!("floating {min_y:.2}")
    } else {
        format!("below ground by {:.2}", -min_y)
    }
}

impl SceneNode {
    /// World-space axis-aligned bounding box of a mesh node: (min, max, size).
    pub fn world_bounds(&self) -> Option<([f32; 3], [f32; 3], [f32; 3])> {
        let mesh = match &self.kind {
            NodeKind::Mesh { mesh, .. } => mesh,
            _ => return None,
        };
        let matrix = self.transform.matrix();
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for point in &mesh.positions {
            let world = matrix.transform_point3(Vec3::from(*point));
            let xyz = [world.x, world.y, world.z];
            for axis in 0..3 {
                min[axis] = min[axis].min(xyz[axis]);
                max[axis] = max[axis].max(xyz[axis]);
            }
        }
        if mesh.positions.is_empty() {
            return None;
        }
        let size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        Some((min, max, size))
    }

    pub fn bounds_json(&self) -> Value {
        match self.world_bounds() {
            Some((min, max, size)) => json!({
                "min": min,
                "max": max,
                "size": size,
                "center": [
                    (min[0] + max[0]) * 0.5,
                    (min[1] + max[1]) * 0.5,
                    (min[2] + max[2]) * 0.5,
                ],
                "ground": ground_status(min[1]),
                "corners": [
                    [min[0], min[1], min[2]], [max[0], min[1], min[2]],
                    [min[0], max[1], min[2]], [max[0], max[1], min[2]],
                    [min[0], min[1], max[2]], [max[0], min[1], max[2]],
                    [min[0], max[1], max[2]], [max[0], max[1], max[2]],
                ],
            }),
            None => Value::Null,
        }
    }

    pub fn to_json(&self) -> Value {
        let base = json!({
            "id": self.id,
            "name": self.name,
            "type": self.kind.label(),
            "parent": self.parent,
            "visible": self.visible,
            "transform": self.transform.to_json(),
        });
        match &self.kind {
            NodeKind::Empty | NodeKind::Camera => base,
            NodeKind::Mesh { mesh, material, primitive } => {
                let mut v = base;
                v["material"] = material.summary_json();
                v["geometry"] = json!({
                    "vertices": mesh.positions.len(),
                    "triangles": mesh.indices.len() / 3,
                    "primitive": primitive.map(|p| p.as_str()),
                });
                v["bounds"] = self.bounds_json();
                v
            }
            NodeKind::Light { light_type, color, intensity } => {
                let mut v = base;
                v["light"] = json!({
                    "type": format!("{:?}", light_type).to_lowercase(),
                    "color": color,
                    "intensity": intensity,
                });
                v
            }
            NodeKind::Armature { armature, pose } => {
                let mut v = base;
                v["armature"] = json!({
                    "bones": armature.bones.iter().enumerate().map(|(index, bone)| json!({
                        "index": index,
                        "name": bone.name,
                        "parent": bone.parent,
                        "length": bone.length,
                        "pose_euler": pose.get(index).map(|t| t.euler()).unwrap_or([0.0; 3]),
                    })).collect::<Vec<_>>(),
                });
                v
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TimelineState {
    pub frame: i32,
    pub frame_start: i32,
    pub frame_end: i32,
    pub fps: u32,
    pub playing: bool,
}

impl Default for TimelineState {
    fn default() -> Self {
        Self { frame: 1, frame_start: 1, frame_end: 250, fps: 24, playing: false }
    }
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub nodes: Vec<SceneNode>,
    pub selected: Option<NodeId>,
    pub next_id: NodeId,
    pub timeline: TimelineState,
    pub background: [f32; 3],
    /// Mesh node id -> skinning data.
    pub skinning: HashMap<NodeId, SkinBinding>,
    /// Bumped on every mutation so the renderer knows when to rebuild GPU resources.
    pub revision: u64,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    pub fn new() -> Self {
        let nodes = vec![
            SceneNode {
                id: 1,
                name: "Camera".to_string(),
                parent: None,
                transform: Transform {
                    position: [4.0, 3.0, 4.0],
                    ..Default::default()
                },
                visible: true,
                kind: NodeKind::Camera,
            },
            SceneNode {
                id: 2,
                name: "Sun".to_string(),
                parent: None,
                transform: Transform {
                    position: [2.0, 5.0, 3.0],
                    ..Default::default()
                },
                visible: true,
                kind: NodeKind::Light {
                    light_type: LightType::Directional,
                    color: [1.0, 0.96, 0.9],
                    intensity: 1.0,
                },
            },
        ];
        Self {
            nodes,
            selected: None,
            next_id: 3,
            timeline: TimelineState::default(),
            background: [0.09, 0.09, 0.11],
            skinning: HashMap::new(),
            revision: 0,
        }
    }

    pub fn node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn node_by_name(&self, name: &str) -> Option<&SceneNode> {
        self.nodes.iter().find(|n| n.name.eq_ignore_ascii_case(name))
    }

    pub fn spawn_primitive(
        &mut self,
        kind: PrimitiveKind,
        name: Option<String>,
        position: Option<[f32; 3]>,
        color: Option<[f32; 4]>,
    ) -> NodeId {
        let mesh = make_primitive(kind);
        let id = self.next_id;
        self.next_id += 1;
        let node = SceneNode {
            id,
            name: name.unwrap_or_else(|| {
                let base = kind.as_str();
                let mut candidate = base.to_string();
                let mut i = 1;
                while self.node_by_name(&candidate).is_some() {
                    i += 1;
                    candidate = format!("{}.{:03}", base, i);
                }
                candidate
            }),
            parent: None,
            transform: Transform { position: position.unwrap_or([0.0; 3]), ..Default::default() },
            visible: true,
            kind: NodeKind::Mesh {
                mesh,
                material: Material {
                    base_color: color.unwrap_or([0.72, 0.74, 0.78, 1.0]),
                    ..Default::default()
                },
                primitive: Some(kind),
            },
        };
        self.nodes.push(node);
        self.selected = Some(id);
        self.revision += 1;
        id
    }

    #[allow(dead_code)] // wired to MCP/tools in a later phase
    pub fn spawn_empty(&mut self, name: Option<String>, position: Option<[f32; 3]>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(SceneNode {
            id,
            name: name.unwrap_or_else(|| format!("Empty.{}", id)),
            parent: None,
            transform: Transform { position: position.unwrap_or([0.0; 3]), ..Default::default() },
            visible: true,
            kind: NodeKind::Empty,
        });
        self.selected = Some(id);
        self.revision += 1;
        id
    }

    #[allow(dead_code)] // wired to MCP/tools in a later phase
    pub fn spawn_light(&mut self, light_type: LightType, name: Option<String>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(SceneNode {
            id,
            name: name.unwrap_or_else(|| format!("{:?}.{}", light_type, id)),
            parent: None,
            transform: Transform::default(),
            visible: true,
            kind: NodeKind::Light { light_type, color: [1.0, 1.0, 1.0], intensity: 1.0 },
        });
        self.selected = Some(id);
        self.revision += 1;
        id
    }

    /// Apply a partial transform update. `position`, `rotation` and `scale`
    /// are only written when `Some`.
    pub fn transform_object(
        &mut self,
        id: NodeId,
        position: Option<[f32; 3]>,
        rotation: Option<[f32; 4]>,
        scale: Option<[f32; 3]>,
    ) -> bool {
        match self.node_mut(id) {
            Some(node) => {
                if let Some(p) = position {
                    node.transform.position = p;
                }
                if let Some(r) = rotation {
                    let q = Quat::from_xyzw(r[0], r[1], r[2], r[3]).normalize();
                    node.transform.rotation = q.to_array();
                }
                if let Some(s) = scale {
                    node.transform.scale = s;
                }
                self.revision += 1;
                true
            }
            None => false,
        }
    }

    pub fn set_node_name(&mut self, id: NodeId, name: String) -> bool {
        match self.node_mut(id) {
            Some(n) => {
                n.name = name;
                self.revision += 1;
                true
            }
            None => false,
        }
    }

    pub fn set_visible(&mut self, id: NodeId, visible: bool) -> bool {
        match self.node_mut(id) {
            Some(n) => {
                n.visible = visible;
                self.revision += 1;
                true
            }
            None => false,
        }
    }

    pub fn set_material_color(&mut self, id: NodeId, color: [f32; 4]) -> bool {
        match self.material_mut(id) {
            Some(material) => {
                material.base_color = color;
                self.revision += 1;
                true
            }
            None => false,
        }
    }

    pub fn material_mut(&mut self, id: NodeId) -> Option<&mut Material> {
        match self.node_mut(id) {
            Some(n) => match &mut n.kind {
                NodeKind::Mesh { material, .. } => Some(material),
                _ => None,
            },
            None => None,
        }
    }

    pub fn mesh_mut(&mut self, id: NodeId) -> Option<&mut MeshData> {
        match self.node_mut(id) {
            Some(n) => match &mut n.kind {
                NodeKind::Mesh { mesh, .. } => Some(mesh),
                _ => None,
            },
            None => None,
        }
    }

    pub fn mesh(&self, id: NodeId) -> Option<&MeshData> {
        match self.node(id) {
            Some(n) => match &n.kind {
                NodeKind::Mesh { mesh, .. } => Some(mesh),
                _ => None,
            },
            None => None,
        }
    }

    /// Add a mesh built from raw vertex/index buffers.
    pub fn create_custom_mesh(
        &mut self,
        name: Option<String>,
        mesh: MeshData,
        material: Option<Material>,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let label = name.unwrap_or_else(|| format!("Mesh.{}", id));
        self.nodes.push(SceneNode {
            id,
            name: label,
            parent: None,
            transform: Transform::default(),
            visible: true,
            kind: NodeKind::Mesh { mesh, material: material.unwrap_or_default(), primitive: None },
        });
        self.selected = Some(id);
        self.revision += 1;
        id
    }

    pub fn delete_node(&mut self, id: NodeId) -> bool {
        if let Some(pos) = self.nodes.iter().position(|n| n.id == id) {
            self.nodes.remove(pos);
            for node in &mut self.nodes {
                if node.parent == Some(id) {
                    node.parent = None;
                }
            }
            if self.selected == Some(id) {
                self.selected = None;
            }
            self.revision += 1;
            true
        } else {
            false
        }
    }

    pub fn create_armature(&mut self, name: Option<String>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let label = name.unwrap_or_else(|| format!("Armature.{}", id));
        self.nodes.push(SceneNode {
            id,
            name: label,
            parent: None,
            transform: Transform::default(),
            visible: true,
            kind: NodeKind::Armature { armature: Armature::new(), pose: Vec::new() },
        });
        self.selected = Some(id);
        self.revision += 1;
        id
    }

    pub fn add_bone(
        &mut self,
        armature_id: NodeId,
        name: String,
        parent: Option<usize>,
        head: [f32; 3],
        tail: [f32; 3],
    ) -> Result<usize, String> {
        let node = self.node_mut(armature_id).ok_or_else(|| "armature not found".to_string())?;
        match &mut node.kind {
            NodeKind::Armature { armature, pose } => {
                let index = armature.add_bone(name, parent, head, tail)?;
                armature.recompute_bind();
                if pose.len() < armature.bones.len() {
                    pose.push(Transform::default());
                }
                self.revision += 1;
                Ok(index)
            }
            _ => Err("node is not an armature".into()),
        }
    }

    pub fn bind_skin(
        &mut self,
        mesh_id: NodeId,
        armature_id: NodeId,
        max_weights: usize,
    ) -> Result<usize, String> {
        let vertices = self
            .mesh(mesh_id)
            .ok_or_else(|| "mesh not found".to_string())?
            .positions
            .clone();
        let armature_node = self.node(armature_id).ok_or_else(|| "armature not found".to_string())?;
        let (joints, weights) = match &armature_node.kind {
            NodeKind::Armature { armature, .. } => armature.auto_weights(&vertices, max_weights),
            _ => return Err("node is not an armature".into()),
        };
        let vertex_count = joints.len();
        self.skinning.insert(mesh_id, SkinBinding { armature: armature_id, joints, weights });
        self.revision += 1;
        Ok(vertex_count)
    }

    pub fn set_bone_pose(
        &mut self,
        armature_id: NodeId,
        bone: usize,
        transform: Transform,
    ) -> Result<(), String> {
        let node = self.node_mut(armature_id).ok_or_else(|| "armature not found".to_string())?;
        match &mut node.kind {
            NodeKind::Armature { pose, .. } => {
                if bone >= pose.len() {
                    return Err(format!("bone index {bone} out of range"));
                }
                pose[bone] = transform;
            }
            _ => return Err("node is not an armature".into()),
        }
        self.revision += 1;
        Ok(())
    }

    pub fn reset_pose(&mut self, armature_id: NodeId) -> Result<usize, String> {
        let node = self.node_mut(armature_id).ok_or_else(|| "armature not found".to_string())?;
        match &mut node.kind {
            NodeKind::Armature { pose, .. } => {
                for entry in pose.iter_mut() {
                    *entry = Transform::default();
                }
                self.revision += 1;
                Ok(pose.len())
            }
            _ => Err("node is not an armature".into()),
        }
    }

    pub fn hierarchy_json(&self) -> Value {
        let nodes: Vec<Value> = self.nodes.iter().map(|n| n.to_json()).collect();
        let mut scene_min = [f32::INFINITY; 3];
        let mut scene_max = [f32::NEG_INFINITY; 3];
        let mut has_bounds = false;
        for node in &self.nodes {
            if let Some((min, max, _)) = node.world_bounds() {
                has_bounds = true;
                for axis in 0..3 {
                    scene_min[axis] = scene_min[axis].min(min[axis]);
                    scene_max[axis] = scene_max[axis].max(max[axis]);
                }
            }
        }
        let scene_bounds = if has_bounds {
            json!({
                "min": scene_min,
                "max": scene_max,
                "size": [scene_max[0] - scene_min[0], scene_max[1] - scene_min[1], scene_max[2] - scene_min[2]],
            })
        } else {
            Value::Null
        };

        let timeline = json!({
            "frame": self.timeline.frame,
            "frame_start": self.timeline.frame_start,
            "frame_end": self.timeline.frame_end,
            "fps": self.timeline.fps,
            "playing": self.timeline.playing,
        });
        json!({
            "scene": {
                "node_count": self.nodes.len(),
                "selected": self.selected,
                "background": self.background,
                "nodes": nodes,
                "bounds": scene_bounds,
                "timeline": timeline.clone(),
            },
            "timeline": timeline,
        })
    }
}

// ---------------------------------------------------------------------------
// Undo / redo: lightweight full-scene snapshots
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct History {
    undo: Vec<Scene>,
    redo: Vec<Scene>,
    limit: usize,
}

impl Default for History {
    fn default() -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), limit: 64 }
    }
}

impl History {
    /// Record the state *before* a mutation.
    pub fn push(&mut self, scene: &Scene) {
        self.undo.push(scene.clone());
        if self.undo.len() > self.limit {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    #[allow(dead_code)]
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    #[allow(dead_code)]
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo(&mut self, current: &mut Scene) -> bool {
        match self.undo.pop() {
            Some(prev) => {
                self.redo.push(current.clone());
                *current = prev;
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self, current: &mut Scene) -> bool {
        match self.redo.pop() {
            Some(next) => {
                self.undo.push(current.clone());
                *current = next;
                true
            }
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Primitive mesh generation (hand-written, no external mesh library)
// ---------------------------------------------------------------------------

fn push_quad(
    out: &mut MeshData,
    normal: [f32; 3],
    corners: [[f32; 3]; 4],
    uv: [[f32; 2]; 4],
) {
    let base = out.positions.len() as u32;
    for i in 0..4 {
        out.positions.push(corners[i]);
        out.normals.push(normal);
        out.uvs.push(uv[i]);
    }
    out.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn mul_add(c: [f32; 3], u: [f32; 3], su: f32, v: [f32; 3], sv: f32) -> [f32; 3] {
    [
        c[0] + u[0] * su + v[0] * sv,
        c[1] + u[1] * su + v[1] * sv,
        c[2] + u[2] * su + v[2] * sv,
    ]
}

fn make_cube() -> MeshData {
    let mut out = MeshData::empty();
    // (normal, u tangent, v tangent) with cross(u, v) == normal
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    let h = 0.5;
    let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    for (n, u, v) in faces {
        // Each face lives on its own side of the cube: centre = normal * half-size.
        let center = [n[0] * h, n[1] * h, n[2] * h];
        let corners = [
            mul_add(center, u, -h, v, -h),
            mul_add(center, u, h, v, -h),
            mul_add(center, u, h, v, h),
            mul_add(center, u, -h, v, h),
        ];
        push_quad(&mut out, n, corners, uv);
    }
    out
}

fn make_plane() -> MeshData {
    let mut out = MeshData::empty();
    let h = 0.5;
    push_quad(
        &mut out,
        [0.0, 1.0, 0.0],
        [
            [-h, 0.0, -h],
            [-h, 0.0, h],
            [h, 0.0, h],
            [h, 0.0, -h],
        ],
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
    );
    out
}

fn make_sphere() -> MeshData {
    let mut out = MeshData::empty();
    let (rings, segments, radius) = (20usize, 32usize, 0.5f32);
    for y in 0..=rings {
        let v = y as f32 / rings as f32;
        let phi = v * std::f32::consts::PI;
        for x in 0..=segments {
            let u = x as f32 / segments as f32;
            let theta = u * TAU;
            let p = [
                phi.sin() * theta.cos() * radius,
                phi.cos() * radius,
                phi.sin() * theta.sin() * radius,
            ];
            let n = Vec3::from(p).normalize();
            out.positions.push(p);
            out.normals.push([n.x, n.y, n.z]);
            out.uvs.push([u, 1.0 - v]);
        }
    }
    let stride = segments as u32 + 1;
    for y in 0..rings as u32 {
        for x in 0..segments as u32 {
            let a = y * stride + x;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            out.indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    out
}

fn make_cylinder() -> MeshData {
    let mut out = MeshData::empty();
    let (segments, radius, height) = (32usize, 0.5f32, 1.0f32);
    let half = height * 0.5;

    // Side wall.
    for i in 0..=segments {
        let t = i as f32 / segments as f32 * TAU;
        let (s, c) = t.sin_cos();
        let n = [c, 0.0, s];
        out.positions.push([c * radius, -half, s * radius]);
        out.normals.push(n);
        out.uvs.push([i as f32 / segments as f32, 0.0]);
        out.positions.push([c * radius, half, s * radius]);
        out.normals.push(n);
        out.uvs.push([i as f32 / segments as f32, 1.0]);
    }
    for i in 0..segments as u32 {
        let a = i * 2; // bottom i
        let c = a + 1; // top i
        let b = a + 2; // bottom i+1
        let d = a + 3; // top i+1
        out.indices.extend_from_slice(&[a, c, b, b, c, d]);
    }

    // Caps.
    for (sign, normal) in [(1.0f32, [0.0f32, 1.0f32, 0.0f32]), (-1.0, [0.0, -1.0, 0.0])] {
        let center = out.positions.len() as u32;
        out.positions.push([0.0, half * sign, 0.0]);
        out.normals.push(normal);
        out.uvs.push([0.5, 0.5]);
        for i in 0..=segments {
            let t = i as f32 / segments as f32 * TAU;
            let (s, c) = t.sin_cos();
            out.positions.push([c * radius, half * sign, s * radius]);
            out.normals.push(normal);
            out.uvs.push([0.5 + c * 0.5, 0.5 + s * 0.5]);
        }
        for i in 0..segments as u32 {
            let a = center + 1 + i;
            let b = center + 1 + i + 1;
            if sign > 0.0 {
                out.indices.extend_from_slice(&[center, b, a]);
            } else {
                out.indices.extend_from_slice(&[center, a, b]);
            }
        }
    }
    out
}

fn make_cone() -> MeshData {
    let mut out = MeshData::empty();
    let (segments, radius, height) = (32usize, 0.5f32, 1.0f32);
    let half = height * 0.5;
    let apex = [0.0, half, 0.0];

    // Flat-shaded side faces: base_i, apex, base_i+1 gives outward winding.
    for i in 0..segments {
        let t0 = i as f32 / segments as f32 * TAU;
        let t1 = (i + 1) as f32 / segments as f32 * TAU;
        let (s0, c0) = t0.sin_cos();
        let (s1, c1) = t1.sin_cos();
        let b0 = [c0 * radius, -half, s0 * radius];
        let b1 = [c1 * radius, -half, s1 * radius];
        let n = (Vec3::from(apex) - Vec3::from(b0))
            .cross(Vec3::from(b1) - Vec3::from(b0))
            .normalize();
        let base = out.positions.len() as u32;
        out.positions.extend_from_slice(&[b0, apex, b1]);
        out.normals.extend_from_slice(&[[n.x, n.y, n.z]; 3]);
        out.uvs.extend_from_slice(&[
            [i as f32 / segments as f32, 0.0],
            [(i as f32 + 0.5) / segments as f32, 1.0],
            [(i + 1) as f32 / segments as f32, 0.0],
        ]);
        out.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    // Base cap (downward normal).
    let center = out.positions.len() as u32;
    out.positions.push([0.0, -half, 0.0]);
    out.normals.push([0.0, -1.0, 0.0]);
    out.uvs.push([0.5, 0.5]);
    for i in 0..=segments {
        let t = i as f32 / segments as f32 * TAU;
        let (s, c) = t.sin_cos();
        out.positions.push([c * radius, -half, s * radius]);
        out.normals.push([0.0, -1.0, 0.0]);
        out.uvs.push([0.5 + c * 0.5, 0.5 + s * 0.5]);
    }
    for i in 0..segments as u32 {
        let a = center + 1 + i;
        let b = center + 1 + i + 1;
        out.indices.extend_from_slice(&[center, a, b]);
    }
    out
}

fn make_torus() -> MeshData {
    let mut out = MeshData::empty();
    let (major_segments, minor_segments) = (48usize, 16usize);
    let (major_r, minor_r) = (0.5f32, 0.18f32);

    for i in 0..=major_segments {
        let theta = i as f32 / major_segments as f32 * TAU;
        let (st, ct) = theta.sin_cos();
        for j in 0..=minor_segments {
            let phi = j as f32 / minor_segments as f32 * TAU;
            let (sp, cp) = phi.sin_cos();
            let ring = major_r + minor_r * cp;
            out.positions.push([ring * ct, minor_r * sp, ring * st]);
            out.normals.push([cp * ct, sp, cp * st]);
            out.uvs.push([i as f32 / major_segments as f32, j as f32 / minor_segments as f32]);
        }
    }
    let stride = minor_segments as u32 + 1;
    for i in 0..major_segments as u32 {
        for j in 0..minor_segments as u32 {
            let a = i * stride + j;
            let b = a + 1; // phi + 1
            let c = a + stride; // theta + 1
            let d = c + 1;
            // cross(phi, theta) is outward for this parameterisation.
            out.indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    out
}

#[cfg(test)]
mod primitive_tests {
    use super::*;

    #[test]
    fn cube_is_a_real_closed_unit_box() {
        let cube = make_cube();
        assert_eq!(cube.positions.len(), 24);
        assert_eq!(cube.normals.len(), 24);
        assert_eq!(cube.indices.len(), 36);
        for axis in 0..3 {
            let min = cube.positions.iter().map(|p| p[axis]).fold(f32::INFINITY, f32::min);
            let max = cube.positions.iter().map(|p| p[axis]).fold(f32::NEG_INFINITY, f32::max);
            assert!((min + 0.5).abs() < 1e-5, "axis {axis} min was {min}");
            assert!((max - 0.5).abs() < 1e-5, "axis {axis} max was {max}");
        }
        // Every vertex must lie on one of the six faces at +/-0.5, not at the origin.
        assert!(cube.positions.iter().all(|p| p.iter().any(|v| v.abs() > 0.49)));
    }
}

pub fn make_primitive(kind: PrimitiveKind) -> MeshData {
    match kind {
        PrimitiveKind::Cube => make_cube(),
        PrimitiveKind::Sphere => make_sphere(),
        PrimitiveKind::Cylinder => make_cylinder(),
        PrimitiveKind::Cone => make_cone(),
        PrimitiveKind::Plane => make_plane(),
        PrimitiveKind::Torus => make_torus(),
    }
}

// ---------------------------------------------------------------------------
// Orbit camera (Blender-style)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { target: Vec3::ZERO, yaw: -0.7, pitch: 0.45, distance: 8.0 }
    }
}

impl OrbitCamera {
    pub fn position(&self) -> Vec3 {
        let cp = self.pitch.cos();
        self.target
            + Vec3::new(
                self.distance * cp * self.yaw.cos(),
                self.distance * self.pitch.sin(),
                self.distance * cp * self.yaw.sin(),
            )
    }

    /// Drag left/right orbits the scene in the same direction as the pointer.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.006;
        self.pitch = (self.pitch - dy * 0.006).clamp(-1.5, 1.5);
    }

    /// Drag pans the scene with the pointer (Blender-style middle/right drag).
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.position()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        let speed = self.distance * 0.0015;
        self.target += right * dx * speed;
        self.target -= up * dy * speed;
    }

    /// Wheel delta can arrive in lines (~1) or pixels (~120). Normalise both,
    /// then apply an exponential step so zooming feels smooth, not snappy.
    pub fn zoom(&mut self, delta: f32) {
        let steps = if delta.abs() > 2.0 { delta / 120.0 } else { delta };
        let factor = (-steps * 0.15).exp();
        self.distance = (self.distance * factor).clamp(0.5, 500.0);
    }

    pub fn focus(&mut self, point: Vec3) {
        self.target = point;
    }
}
