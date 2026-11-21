//! Skeletal rigs: bones, bind matrices, posing and linear-blend skinning.
//!
//! This module is deliberately free of rendering APIs. It works with the
//! [`Transform`] type from [`crate::scene`] and `glam` math types only.

use crate::scene::Transform;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Serde bridge for `glam::Mat4`.
///
/// The crate's `glam` dependency is built without glam's optional `serde`
/// feature, so `Mat4` does not implement `Serialize`/`Deserialize` on its own.
/// This module stores the matrix compactly as its 16 column-major floats and
/// lets `Bone` keep the requested derives and `glam::Mat4` field type.
mod mat4_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(matrix: &glam::Mat4, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Serialize::serialize(&matrix.to_cols_array(), serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<glam::Mat4, D::Error>
    where
        D: Deserializer<'de>,
    {
        let cols = <[f32; 16]>::deserialize(deserializer)?;
        Ok(glam::Mat4::from_cols_array(&cols))
    }
}

/// A single bone in an [`Armature`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bone {
    pub name: String,
    pub parent: Option<usize>,
    /// Rest transform relative to the parent bone. `position` is the head
    /// position relative to the parent head (expressed in the parent's local
    /// frame); `rotation` orients the bone's local +Y along its length.
    pub rest: Transform,
    /// Bone length along local +Y.
    pub length: f32,
    /// Inverse bind matrix (computed by [`Armature::recompute_bind`]).
    #[serde(with = "mat4_serde")]
    pub inverse_bind: glam::Mat4,
}

/// A hierarchy of bones. Parent indices refer to positions in `bones`.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Armature {
    pub bones: Vec<Bone>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Global rest matrix of every bone, computed as `parent_global * local`.
/// Memoised recursion handles parent indices in any order; malformed cycles
/// are cut off by the depth guard.
fn global_rest_matrices(bones: &[Bone]) -> Vec<Mat4> {
    let n = bones.len();
    let mut out = vec![Mat4::IDENTITY; n];
    let mut done = vec![false; n];
    for i in 0..n {
        resolve_rest(bones, i, &mut out, &mut done, 0);
    }
    out
}

fn resolve_rest(bones: &[Bone], i: usize, out: &mut [Mat4], done: &mut [bool], depth: usize) {
    if done[i] || depth > bones.len() {
        return;
    }
    let local = bones[i].rest.matrix();
    let global = match bones[i].parent {
        Some(p) if p < bones.len() && p != i => {
            resolve_rest(bones, p, out, done, depth + 1);
            out[p] * local
        }
        _ => local,
    };
    out[i] = global;
    done[i] = true;
}

/// Global pose matrix of every bone, using a pose entry when present and the
/// bone's rest transform otherwise.
fn pose_global_matrices(bones: &[Bone], pose: &[Transform]) -> Vec<Mat4> {
    let n = bones.len();
    let mut out = vec![Mat4::IDENTITY; n];
    let mut done = vec![false; n];
    for i in 0..n {
        resolve_pose(bones, pose, i, &mut out, &mut done, 0);
    }
    out
}

fn resolve_pose(
    bones: &[Bone],
    pose: &[Transform],
    i: usize,
    out: &mut [Mat4],
    done: &mut [bool],
    depth: usize,
) {
    if done[i] || depth > bones.len() {
        return;
    }
    let local = pose.get(i).unwrap_or(&bones[i].rest).matrix();
    let global = match bones[i].parent {
        Some(p) if p < bones.len() && p != i => {
            resolve_pose(bones, pose, p, out, done, depth + 1);
            out[p] * local
        }
        _ => local,
    };
    out[i] = global;
    done[i] = true;
}

/// Squared-distance-friendly distance from a point to a line segment.
fn distance_to_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let denom = ab.length_squared();
    let t = if denom > 1e-12 {
        ((p - a).dot(ab) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + ab * t)).length()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl Armature {
    pub fn new() -> Self {
        Self { bones: Vec::new() }
    }

    /// Add a bone specified by world-space head/tail at bind time. The parent
    /// index must refer to an existing bone; the head is stored relative to
    /// the parent's head, in the parent's local frame.
    pub fn add_bone(
        &mut self,
        name: String,
        parent: Option<usize>,
        head: [f32; 3],
        tail: [f32; 3],
    ) -> Result<usize, String> {
        if let Some(p) = parent {
            if p >= self.bones.len() {
                return Err(format!(
                    "parent index {p} is out of range for {} bones",
                    self.bones.len()
                ));
            }
        }

        let globals = global_rest_matrices(&self.bones);
        let parent_inv = parent.map(|p| globals[p].inverse()).unwrap_or(Mat4::IDENTITY);

        let head_v = Vec3::from(head);
        let world_dir = Vec3::from(tail) - head_v;
        let length = world_dir.length();

        // Head offset in the parent's local frame == rest.position.
        let local_head = parent_inv.transform_point3(head_v);

        // Orientation in the parent's local frame: rotate local +Y onto the
        // bone direction. A zero-length bone keeps the identity rotation.
        let rotation = if length > 1e-8 {
            let local_dir = parent_inv.transform_vector3(world_dir).normalize();
            Quat::from_rotation_arc(Vec3::Y, local_dir)
        } else {
            Quat::IDENTITY
        };

        let index = self.bones.len();
        self.bones.push(Bone {
            name,
            parent,
            rest: Transform {
                position: [local_head.x, local_head.y, local_head.z],
                rotation: rotation.to_array(),
                scale: [1.0; 3],
            },
            length,
            inverse_bind: Mat4::IDENTITY, // assigned by recompute_bind
        });
        Ok(index)
    }

    /// Recompute every bone's inverse bind matrix from the rest hierarchy
    /// (the armature object is assumed to be at the origin).
    pub fn recompute_bind(&mut self) {
        let globals = global_rest_matrices(&self.bones);
        for (bone, global) in self.bones.iter_mut().zip(globals) {
            bone.inverse_bind = global.inverse();
        }
    }

    /// Global pose matrix for each bone. Pose entries are local transforms;
    /// missing entries fall back to the bone's rest transform.
    pub fn pose_matrices(&self, pose: &[Transform]) -> Vec<Mat4> {
        pose_global_matrices(&self.bones, pose)
    }

    /// World-space head and tail of a bone under a pose.
    pub fn bone_head_tail(&self, index: usize, pose: &[Transform]) -> (Vec3, Vec3) {
        if index >= self.bones.len() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let matrices = self.pose_matrices(pose);
        let m = matrices[index];
        let head = m.transform_point3(Vec3::ZERO);
        let tail = m.transform_point3(Vec3::new(0.0, self.bones[index].length, 0.0));
        (head, tail)
    }

    /// Distance-based envelope auto weights. Returns parallel
    /// `(joints, weights)` arrays with four slots per vertex.
    pub fn auto_weights(
        &self,
        vertices: &[[f32; 3]],
        max_weights: usize,
    ) -> (Vec<[u16; 4]>, Vec<[f32; 4]>) {
        let keep = max_weights.min(4);
        let mut joints = Vec::with_capacity(vertices.len());
        let mut weights = Vec::with_capacity(vertices.len());

        if self.bones.is_empty() || keep == 0 {
            for _ in vertices {
                joints.push([0u16; 4]);
                weights.push([0.0f32; 4]);
            }
            return (joints, weights);
        }

        // Envelope segments in their rest (bind) position.
        let segments: Vec<(Vec3, Vec3)> =
            (0..self.bones.len()).map(|i| self.bone_head_tail(i, &[])).collect();

        for vertex in vertices {
            let p = Vec3::from(*vertex);
            let mut candidates: Vec<(usize, f32)> = segments
                .iter()
                .enumerate()
                .map(|(i, (a, b))| {
                    let d = distance_to_segment(p, *a, *b);
                    (i, 1.0 / (d * d + 0.0001))
                })
                .collect();
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(keep);

            // Drop negligible influences, but never end up with an empty set.
            let filtered: Vec<(usize, f32)> = candidates
                .iter()
                .copied()
                .filter(|(_, w)| *w >= 0.001)
                .collect();
            let kept = if filtered.is_empty() && !candidates.is_empty() {
                vec![candidates[0]]
            } else {
                filtered
            };

            let sum: f32 = kept.iter().map(|(_, w)| *w).sum();
            let mut j = [0u16; 4];
            let mut w = [0.0f32; 4];
            if sum > 0.0 {
                for (slot, (index, weight)) in kept.iter().take(4).enumerate() {
                    j[slot] = *index as u16;
                    w[slot] = weight / sum;
                }
            }
            joints.push(j);
            weights.push(w);
        }

        (joints, weights)
    }
}

/// Linearly blend up to four bone transforms.
///
/// `pose_matrices` and `inverse_binds` are indexed by joint id. The function
/// accumulates `weight * (pose * inverse_bind).transform_point3(position)` and
/// returns the position unchanged when the total weight is zero.
pub fn skin_vertex(
    position: Vec3,
    joints: [u16; 4],
    weights: [f32; 4],
    pose_matrices: &[Mat4],
    inverse_binds: &[Mat4],
) -> Vec3 {
    let mut result = Vec3::ZERO;
    let mut total = 0.0f32;
    for i in 0..4 {
        let weight = weights[i];
        if weight == 0.0 {
            continue;
        }
        let joint = joints[i] as usize;
        if joint >= pose_matrices.len() || joint >= inverse_binds.len() {
            continue;
        }
        let skin = pose_matrices[joint] * inverse_binds[joint];
        result += skin.transform_point3(position) * weight;
        total += weight;
    }
    if total.abs() <= 1e-12 {
        position
    } else {
        // Blender divides the weighted delta by the sum of weights at deform time.
        result / total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a - b).length() < eps
    }

    fn quat_from(a: [f32; 4]) -> Quat {
        Quat::from_xyzw(a[0], a[1], a[2], a[3])
    }

    #[test]
    fn add_bone_chain_inverse_bind() {
        let mut arm = Armature::new();
        let root = arm.add_bone("root".into(), None, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
        let mid = arm.add_bone("mid".into(), Some(root), [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]).unwrap();
        let tip = arm.add_bone("tip".into(), Some(mid), [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]).unwrap();
        assert_eq!((root, mid, tip), (0, 1, 2));

        // Rest heads are relative to the parent's head.
        assert!(close(Vec3::from(arm.bones[mid].rest.position), Vec3::new(0.0, 1.0, 0.0), 1e-6));
        assert!(close(Vec3::from(arm.bones[tip].rest.position), Vec3::new(0.0, 1.0, 0.0), 1e-6));

        // The tip's rest rotation maps local +Y onto world +X.
        assert!(close(quat_from(arm.bones[tip].rest.rotation) * Vec3::Y, Vec3::X, 1e-5));
        assert!((arm.bones[tip].length - 1.0).abs() < 1e-6);

        arm.recompute_bind();

        // Inverse bind of the tip maps its world tail to its local tail.
        let local_tail = arm.bones[tip].inverse_bind.transform_point3(Vec3::new(1.0, 2.0, 0.0));
        assert!(close(local_tail, Vec3::new(0.0, arm.bones[tip].length, 0.0), 1e-5));

        // Inverse bind of the root maps its world head to the local origin.
        let local_head = arm.bones[root].inverse_bind.transform_point3(Vec3::ZERO);
        assert!(close(local_head, Vec3::ZERO, 1e-6));

        // Invalid parents are rejected without mutating the armature.
        assert!(arm.add_bone("bad".into(), Some(99), [0.0; 3], [0.0; 3]).is_err());
        assert_eq!(arm.bones.len(), 3);
    }

    #[test]
    fn pose_matrices_inherit_parent() {
        let mut arm = Armature::new();
        let root = arm.add_bone("root".into(), None, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
        let child = arm.add_bone("child".into(), Some(root), [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]).unwrap();
        arm.recompute_bind();

        assert!(close(arm.bone_head_tail(child, &[]).0, Vec3::new(0.0, 1.0, 0.0), 1e-6));

        let mut root_pose = Transform::default();
        root_pose.rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2).to_array();
        // Only the root is supplied; the child falls back to its rest transform.
        let pose = vec![root_pose];
        let matrices = arm.pose_matrices(&pose);

        let root_tail = matrices[root].transform_point3(Vec3::new(0.0, 1.0, 0.0));
        assert!(close(root_tail, Vec3::new(-1.0, 0.0, 0.0), 1e-5));

        let child_head = matrices[child].transform_point3(Vec3::ZERO);
        assert!(close(child_head, Vec3::new(-1.0, 0.0, 0.0), 1e-5), "child head: {child_head}");
    }

    #[test]
    fn bone_head_tail_respects_length() {
        let mut arm = Armature::new();
        let index = arm.add_bone("leg".into(), None, [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]).unwrap();

        let (head, tail) = arm.bone_head_tail(index, &[]);
        assert!(close(head, Vec3::ZERO, 1e-6));
        assert!(close(tail, Vec3::new(0.0, 3.0, 0.0), 1e-6));
        assert!(((tail - head).length() - 3.0).abs() < 1e-6);

        let mut pose = Transform::default();
        pose.rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2).to_array();
        let (head2, tail2) = arm.bone_head_tail(index, &[pose]);
        assert!(close(head2, Vec3::ZERO, 1e-6));
        assert!(((tail2 - head2).length() - 3.0).abs() < 1e-6);
        assert!(close(tail2, Vec3::new(-3.0, 0.0, 0.0), 1e-5));

        // Out-of-range indices are safe.
        assert_eq!(arm.bone_head_tail(99, &[]), (Vec3::ZERO, Vec3::ZERO));
    }

    #[test]
    fn auto_weights_favour_nearby_bone() {
        let mut arm = Armature::new();
        let near = arm.add_bone("near".into(), None, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
        let far = arm.add_bone("far".into(), None, [10.0, 0.0, 0.0], [10.0, 1.0, 0.0]).unwrap();

        let vertices: Vec<[f32; 3]> = vec![[0.0, 0.5, 0.0], [10.0, 0.5, 0.0]];
        let (joints, weights) = arm.auto_weights(&vertices, 4);
        assert_eq!(joints.len(), 2);
        assert_eq!(weights.len(), 2);

        assert_eq!(joints[0][0], near as u16);
        assert!(weights[0][0] > 0.99, "weights: {:?}", weights[0]);
        assert!((weights[0].iter().sum::<f32>() - 1.0).abs() < 1e-5);

        assert_eq!(joints[1][0], far as u16);
        assert!(weights[1][0] > 0.99, "weights: {:?}", weights[1]);
        assert!((weights[1].iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn skin_vertex_rotates_with_bone() {
        let mut arm = Armature::new();
        let bone = arm.add_bone("spin".into(), None, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
        arm.recompute_bind();

        let mut pose = Transform::default();
        pose.rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2).to_array();
        let matrices = arm.pose_matrices(&[pose]);
        let binds: Vec<Mat4> = arm.bones.iter().map(|b| b.inverse_bind).collect();

        let point = Vec3::new(0.0, 1.0, 0.0);
        let moved = skin_vertex(point, [bone as u16, 0, 0, 0], [1.0, 0.0, 0.0, 0.0], &matrices, &binds);
        assert!(close(moved, Vec3::new(-1.0, 0.0, 0.0), 1e-5), "moved: {moved}");

        // Zero total weight leaves the vertex untouched.
        let unchanged = skin_vertex(point, [0, 0, 0, 0], [0.0; 4], &matrices, &binds);
        assert!(close(unchanged, point, 1e-8));
    }
}
