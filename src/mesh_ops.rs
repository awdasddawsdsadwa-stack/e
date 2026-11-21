//! CPU-side mesh operations used by the editor's "modify mesh" tools.
//!
//! Everything here works on a plain [`MeshData`] so it can be exercised in
//! unit tests without a rendering context. All math uses `glam::Vec3`.

use std::collections::HashMap;

use glam::Vec3;

use crate::scene::MeshData;

/// Safe upper bound for a single midpoint-subdivision request.
const MAX_SUBDIV_LEVELS: u32 = 3;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn vec3(a: [f32; 3]) -> Vec3 {
    Vec3::from(a)
}

fn array3(v: Vec3) -> [f32; 3] {
    v.to_array()
}

/// Normalize `v`, falling back to `fallback` for zero / non-finite vectors.
fn normalized_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let a = v.to_array();
    if v.length_squared() > 1e-12 && a.iter().all(|x| x.is_finite()) {
        v.normalize()
    } else {
        fallback
    }
}

/// Normals guaranteed to be aligned with `positions`.
fn normals_of(mesh: &MeshData) -> Vec<[f32; 3]> {
    if mesh.normals.len() == mesh.positions.len() {
        mesh.normals.clone()
    } else {
        vec![[0.0, 1.0, 0.0]; mesh.positions.len()]
    }
}

/// UVs guaranteed to be aligned with `positions`.
fn uvs_of(mesh: &MeshData) -> Vec<[f32; 2]> {
    if mesh.uvs.len() == mesh.positions.len() {
        mesh.uvs.clone()
    } else {
        vec![[0.0, 0.0]; mesh.positions.len()]
    }
}

/// Geometric face normal for a triangle, with a safe fallback.
fn face_normal(positions: &[[f32; 3]], tri: &[u32; 3]) -> Vec3 {
    let (Some(&a), Some(&b), Some(&c)) = (
        positions.get(tri[0] as usize),
        positions.get(tri[1] as usize),
        positions.get(tri[2] as usize),
    ) else {
        return Vec3::Y;
    };
    normalized_or((vec3(b) - vec3(a)).cross(vec3(c) - vec3(a)), Vec3::Y)
}

// ---------------------------------------------------------------------------
// Public operations
// ---------------------------------------------------------------------------

/// Auto-UV box projection.
///
/// For each vertex the dominant axis of its normal is selected and the other
/// two position coordinates are projected onto the UV plane, scaled by
/// `scale` (use `1.0` for world-space UVs). The sign of the dominant normal
/// component flips one axis so adjacent faces keep a consistent, non-mirrored
/// orientation. If the normal buffer is missing or mismatched, a face normal
/// is derived from the first triangle referencing that vertex.
pub fn box_project_uvs(mesh: &mut MeshData, scale: f32) {
    if mesh.positions.is_empty() {
        mesh.uvs.clear();
        return;
    }

    let have_normals = mesh.normals.len() == mesh.positions.len();
    let inv_scale = if scale.is_finite() && scale.abs() > 1e-12 {
        1.0 / scale
    } else {
        1.0
    };

    let mut uvs = Vec::with_capacity(mesh.positions.len());
    for (i, &p) in mesh.positions.iter().enumerate() {
        let n = if have_normals {
            let n = vec3(mesh.normals[i]);
            normalized_or(n, Vec3::Y)
        } else {
            let mut derived = None;
            for tri in mesh.indices.chunks_exact(3) {
                let ids = [tri[0], tri[1], tri[2]];
                if ids.contains(&(i as u32)) {
                    derived = Some(face_normal(&mesh.positions, &ids));
                    break;
                }
            }
            derived.unwrap_or(Vec3::Y)
        };

        let a = n.abs().to_array();
        let axis = if a[0] >= a[1] && a[0] >= a[2] {
            0
        } else if a[1] >= a[2] {
            1
        } else {
            2
        };

        let uv = match axis {
            0 => {
                // Dominant X: project (z, y); flip u on the -X face.
                let u = if n.x >= 0.0 { -p[2] } else { p[2] };
                [u * inv_scale, p[1] * inv_scale]
            }
            1 => {
                // Dominant Y: project (x, z); flip v on the -Y face.
                let v = if n.y >= 0.0 { -p[2] } else { p[2] };
                [p[0] * inv_scale, v * inv_scale]
            }
            _ => {
                // Dominant Z: project (x, y); flip u on the -Z face.
                let u = if n.z >= 0.0 { p[0] } else { -p[0] };
                [u * inv_scale, p[1] * inv_scale]
            }
        };
        uvs.push(uv);
    }
    mesh.uvs = uvs;
}

/// Append a midpoint vertex with averaged normal and interpolated UV.
#[allow(clippy::too_many_arguments)]
fn push_midpoint(
    out_positions: &mut Vec<[f32; 3]>,
    out_normals: &mut Vec<[f32; 3]>,
    out_uvs: &mut Vec<[f32; 2]>,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    a: usize,
    b: usize,
) -> u32 {
    let pm = (vec3(positions[a]) + vec3(positions[b])) * 0.5;
    let nm = normalized_or(vec3(normals[a]) + vec3(normals[b]), vec3(normals[a]));
    let uv = [
        (uvs[a][0] + uvs[b][0]) * 0.5,
        (uvs[a][1] + uvs[b][1]) * 0.5,
    ];
    let index = out_positions.len() as u32;
    out_positions.push(array3(pm));
    out_normals.push(array3(nm));
    out_uvs.push(uv);
    index
}

/// Midpoint subdivision: every triangle becomes four.
///
/// Normals at the new vertices are averaged and normalized, UVs are linearly
/// interpolated. `levels` is clamped to [`MAX_SUBDIV_LEVELS`].
pub fn subdivide(mesh: &mut MeshData, levels: u32) -> Result<(), String> {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        return Err("Cannot subdivide an empty mesh".to_string());
    }
    if mesh.indices.len() % 3 != 0 {
        return Err("Index buffer length is not a multiple of 3".to_string());
    }
    for &idx in &mesh.indices {
        if idx as usize >= mesh.positions.len() {
            return Err(format!(
                "Index {idx} is out of range for {} vertices",
                mesh.positions.len()
            ));
        }
    }
    for tri in mesh.indices.chunks_exact(3) {
        let a = vec3(mesh.positions[tri[0] as usize]);
        let b = vec3(mesh.positions[tri[1] as usize]);
        let c = vec3(mesh.positions[tri[2] as usize]);
        if (b - a).cross(c - a).length_squared() <= 1e-18 {
            return Err("Mesh contains a degenerate (zero-area) triangle".to_string());
        }
    }

    let levels = levels.min(MAX_SUBDIV_LEVELS);
    for _ in 0..levels {
        let positions = mesh.positions.clone();
        let normals = normals_of(mesh);
        let uvs = uvs_of(mesh);
        let old_indices = mesh.indices.clone();

        let mut new_positions = positions.clone();
        let mut new_normals = normals.clone();
        let mut new_uvs = uvs.clone();
        let mut new_indices = Vec::with_capacity(old_indices.len() * 4);

        for tri in old_indices.chunks_exact(3) {
            let ids = [tri[0], tri[1], tri[2]];
            let ia = ids[0] as usize;
            let ib = ids[1] as usize;
            let ic = ids[2] as usize;

            let mab = push_midpoint(
                &mut new_positions,
                &mut new_normals,
                &mut new_uvs,
                &positions,
                &normals,
                &uvs,
                ia,
                ib,
            );
            let mbc = push_midpoint(
                &mut new_positions,
                &mut new_normals,
                &mut new_uvs,
                &positions,
                &normals,
                &uvs,
                ib,
                ic,
            );
            let mca = push_midpoint(
                &mut new_positions,
                &mut new_normals,
                &mut new_uvs,
                &positions,
                &normals,
                &uvs,
                ic,
                ia,
            );

            // Four CCW triangles, preserving the original winding.
            new_indices.extend_from_slice(&[
                ids[0], mab, mca, //
                mab, ids[1], mbc, //
                mca, mbc, ids[2], //
                mab, mbc, mca,
            ]);
        }

        mesh.positions = new_positions;
        mesh.normals = new_normals;
        mesh.uvs = new_uvs;
        mesh.indices = new_indices;
    }
    Ok(())
}

/// Extrude a mesh along `direction` by `distance`.
///
/// The original and translated copies keep their normals/UVs. Edges that
/// belong to exactly one triangle are stitched with quads that get their own
/// flat-shaded side-wall vertices and normals.
pub fn extrude(mesh: &mut MeshData, direction: [f32; 3], distance: f32) -> Result<(), String> {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        return Err("Cannot extrude an empty mesh".to_string());
    }
    if mesh.indices.len() % 3 != 0 {
        return Err("Index buffer length is not a multiple of 3".to_string());
    }
    let dir = vec3(direction);
    if dir.length_squared() <= 1e-12 || dir.to_array().iter().any(|x| !x.is_finite()) {
        return Err("Extrusion direction must be a non-zero, finite vector".to_string());
    }
    if !distance.is_finite() {
        return Err("Extrusion distance must be finite".to_string());
    }
    let offset = dir.normalize() * distance;

    let base_v = mesh.positions.len();
    let old_normals = normals_of(mesh);
    let old_uvs = uvs_of(mesh);
    let old_indices = mesh.indices.clone();

    let mut positions = Vec::with_capacity(base_v * 2);
    let mut normals = Vec::with_capacity(base_v * 2);
    let mut uvs = Vec::with_capacity(base_v * 2);
    let mut indices = Vec::with_capacity(old_indices.len() * 2);

    // Bottom copy (unchanged).
    positions.extend_from_slice(&mesh.positions);
    normals.extend_from_slice(&old_normals);
    uvs.extend_from_slice(&old_uvs);
    indices.extend_from_slice(&old_indices);

    // Top copy (translated; original normals and UVs preserved).
    let base_u32 = base_v as u32;
    for &p in &mesh.positions {
        positions.push(array3(vec3(p) + offset));
    }
    normals.extend_from_slice(&old_normals);
    uvs.extend_from_slice(&old_uvs);
    indices.extend(old_indices.iter().map(|&i| i + base_u32));

    // Find boundary edges: undirected edges used by exactly one triangle.
    struct EdgeInfo {
        count: usize,
        a: u32,
        b: u32,
        face: Vec3,
    }
    let mut edges: HashMap<(u32, u32), EdgeInfo> = HashMap::new();
    for tri in old_indices.chunks_exact(3) {
        let ids = [tri[0], tri[1], tri[2]];
        let face = face_normal(&mesh.positions, &ids);
        for &(u, v) in &[(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
            let key = if u <= v { (u, v) } else { (v, u) };
            let entry = edges.entry(key).or_insert(EdgeInfo {
                count: 0,
                a: u,
                b: v,
                face,
            });
            entry.count += 1;
        }
    }

    // Sort for deterministic output.
    let mut sorted: Vec<_> = edges.into_iter().collect();
    sorted.sort_by_key(|(key, _)| *key);

    for (_, info) in sorted {
        if info.count != 1 {
            continue;
        }
        let va = vec3(mesh.positions[info.a as usize]);
        let vb = vec3(mesh.positions[info.b as usize]);
        let flip = info.face.dot(offset) < 0.0;
        let mut side = normalized_or((vb - va).cross(offset), Vec3::Y);
        if flip {
            side = -side;
        }
        let flat = array3(side);

        // Four fresh vertices so the wall normal stays flat.
        let start = positions.len() as u32;
        positions.extend_from_slice(&[
            array3(va),
            array3(vb),
            array3(va + offset),
            array3(vb + offset),
        ]);
        normals.extend_from_slice(&[flat; 4]);
        let len = (vb - va).length();
        let h = distance.abs();
        uvs.extend_from_slice(&[[0.0, 0.0], [len, 0.0], [0.0, h], [len, h]]);

        if flip {
            indices.extend_from_slice(&[start, start + 3, start + 1, start, start + 2, start + 3]);
        } else {
            indices.extend_from_slice(&[start, start + 1, start + 3, start, start + 3, start + 2]);
        }
    }

    *mesh = MeshData {
        positions,
        normals,
        uvs,
        indices,
    };
    Ok(())
}

/// Append `other` to `target`, offsetting its indices by the target's vertex
/// count. Missing normals/UVs on either side are filled with safe defaults.
pub fn merge(target: &mut MeshData, other: &MeshData) {
    let base_v = target.positions.len();
    let other_v = other.positions.len();
    let offset = base_v as u32;

    let mut normals = Vec::with_capacity(base_v + other_v);
    if target.normals.len() == base_v {
        normals.extend_from_slice(&target.normals);
    } else {
        normals.extend(vec![[0.0, 1.0, 0.0]; base_v]);
    }
    if other.normals.len() == other_v {
        normals.extend_from_slice(&other.normals);
    } else {
        normals.extend(vec![[0.0, 1.0, 0.0]; other_v]);
    }

    let mut uvs = Vec::with_capacity(base_v + other_v);
    if target.uvs.len() == base_v {
        uvs.extend_from_slice(&target.uvs);
    } else {
        uvs.extend(vec![[0.0, 0.0]; base_v]);
    }
    if other.uvs.len() == other_v {
        uvs.extend_from_slice(&other.uvs);
    } else {
        uvs.extend(vec![[0.0, 0.0]; other_v]);
    }

    target.positions.extend_from_slice(&other.positions);
    target
        .indices
        .extend(other.indices.iter().map(|&i| i + offset));
    target.normals = normals;
    target.uvs = uvs;
}

/// Remove triangles by their index into the triangle list (`indices / 3`).
///
/// Vertices are deliberately left in place, so they may become unreferenced.
pub fn delete_faces(mesh: &mut MeshData, face_indices: &[usize]) -> Result<(), String> {
    let face_count = mesh.indices.len() / 3;
    for &f in face_indices {
        if f >= face_count {
            return Err(format!(
                "Face index {f} out of range (mesh has {face_count} triangles)"
            ));
        }
    }

    let mut removed = vec![false; face_count];
    for &f in face_indices {
        removed[f] = true;
    }

    mesh.indices = mesh
        .indices
        .chunks_exact(3)
        .enumerate()
        .filter(|(i, _)| !removed[*i])
        .flat_map(|(_, tri)| tri.iter().copied())
        .collect();
    Ok(())
}

/// Thin wrapper around [`MeshData::recalculate_normals`].
pub fn recalculate_normals(mesh: &mut MeshData) {
    mesh.recalculate_normals();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A single triangle in the XY plane, wound counter-clockwise (+Z).
    fn triangle() -> MeshData {
        MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
        }
    }

    /// A quad in the XY plane (two triangles, four shared vertices).
    fn quad() -> MeshData {
        MeshData {
            positions: vec![
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            indices: vec![0, 1, 2, 0, 2, 3],
        }
    }

    #[test]
    fn box_project_uvs_projects_dominant_axis() {
        let mut mesh = quad();
        mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
        box_project_uvs(&mut mesh, 1.0);
        assert_eq!(mesh.uvs.len(), mesh.positions.len());
        assert_eq!(mesh.uvs, vec![[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]);

        // Scale divides the projected coordinates.
        let mut mesh = quad();
        box_project_uvs(&mut mesh, 2.0);
        assert_eq!(mesh.uvs.len(), 4);
        assert!((mesh.uvs[1][0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn box_project_uvs_derives_normals_when_missing() {
        let mut mesh = triangle();
        mesh.normals.clear();
        box_project_uvs(&mut mesh, 1.0);
        assert_eq!(mesh.uvs.len(), mesh.positions.len());
        assert_eq!(mesh.uvs[0], [0.0, 0.0]);
        assert_eq!(mesh.uvs[1], [1.0, 0.0]);
    }

    #[test]
    fn subdivide_quad_counts() {
        let mut mesh = quad();
        subdivide(&mut mesh, 1).unwrap();
        // 4 vertices + 2 triangles * 3 midpoints; 2 triangles * 4 = 8 triangles.
        assert_eq!(mesh.positions.len(), 10);
        assert_eq!(mesh.indices.len(), 24);
        assert_eq!(mesh.normals.len(), 10);
        assert_eq!(mesh.uvs.len(), 10);
    }

    #[test]
    fn subdivide_levels_are_clamped() {
        let mut a = quad();
        let mut b = quad();
        subdivide(&mut a, 99).unwrap();
        subdivide(&mut b, MAX_SUBDIV_LEVELS).unwrap();
        assert_eq!(a.positions.len(), b.positions.len());
        assert_eq!(a.indices.len(), b.indices.len());
        // levels == 0 is a no-op.
        let mut c = quad();
        subdivide(&mut c, 0).unwrap();
        assert_eq!(c.indices, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn subdivide_rejects_empty_and_degenerate() {
        let mut empty = MeshData::empty();
        assert!(subdivide(&mut empty, 1).is_err());
        let mut bad = triangle();
        bad.positions[2] = [2.0, 0.0, 0.0]; // collinear => zero area
        assert!(subdivide(&mut bad, 1).is_err());
    }

    #[test]
    fn merge_appends_and_offsets() {
        let mut target = quad();
        let other = quad();
        merge(&mut target, &other);
        assert_eq!(target.positions.len(), 8);
        assert_eq!(target.indices.len(), 12);
        assert_eq!(target.normals.len(), 8);
        assert_eq!(target.uvs.len(), 8);
        assert_eq!(&target.indices[6..], &[4, 5, 6, 4, 6, 7]);
    }

    #[test]
    fn merge_fills_missing_normals_and_uvs() {
        let mut target = triangle();
        target.normals.clear();
        target.uvs.clear();
        let mut other = triangle();
        other.normals.clear();
        other.uvs.clear();
        merge(&mut target, &other);
        assert_eq!(target.positions.len(), 6);
        assert_eq!(target.indices.len(), 6);
        assert_eq!(target.normals, vec![[0.0, 1.0, 0.0]; 6]);
        assert_eq!(target.uvs, vec![[0.0, 0.0]; 6]);
    }

    #[test]
    fn delete_faces_removes_selected_triangles() {
        let mut mesh = quad();
        delete_faces(&mut mesh, &[0]).unwrap();
        assert_eq!(mesh.indices, vec![0, 2, 3]);
        assert_eq!(mesh.positions.len(), 4); // unreferenced vertices kept
        assert!(delete_faces(&mut mesh, &[1]).is_err());
        assert!(delete_faces(&mut mesh, &[0, 0]).is_ok()); // duplicates are fine
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn extrude_triangle_counts() {
        let mut mesh = triangle();
        extrude(&mut mesh, [0.0, 0.0, 2.0], 3.0).unwrap();
        // 2 copies * 3 vertices + 3 boundary edges * 4 wall vertices = 18.
        assert_eq!(mesh.positions.len(), 18);
        // 2 original triangles + 3 walls * 2 triangles = 8 triangles.
        assert_eq!(mesh.indices.len(), 24);
        assert_eq!(mesh.normals.len(), 18);
        assert_eq!(mesh.uvs.len(), 18);
    }

    #[test]
    fn extrude_quad_counts() {
        let mut mesh = quad();
        extrude(&mut mesh, [0.0, 1.0, 0.0], 1.5).unwrap();
        // 2 * 4 + 4 boundary edges * 4 = 24 vertices, 12 triangles.
        assert_eq!(mesh.positions.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
        assert_eq!(mesh.normals.len(), 24);
        assert_eq!(mesh.uvs.len(), 24);
    }

    #[test]
    fn extrude_rejects_invalid_input() {
        let mut empty = MeshData::empty();
        assert!(extrude(&mut empty, [0.0, 0.0, 1.0], 1.0).is_err());
        let mut mesh = triangle();
        assert!(extrude(&mut mesh, [0.0, 0.0, 0.0], 1.0).is_err());
        assert!(extrude(&mut mesh, [f32::NAN, 0.0, 0.0], 1.0).is_err());
    }
}
