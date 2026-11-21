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

/// Compute smooth (area-weighted) vertex normals directly from a position /
/// index buffer pair.
///
/// This is deliberately self-contained: it does not call any helper in
/// `scene.rs`. Each triangle contributes its un-normalised geometric normal
/// (so larger triangles weigh more) to each of its three corners, and the
/// accumulated sums are normalised at the end. Zero or non-finite sums fall
/// back to `+Y`.
fn compute_smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut sums = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if a >= sums.len() || b >= sums.len() || c >= sums.len() {
            continue;
        }
        let n = (vec3(positions[b]) - vec3(positions[a]))
            .cross(vec3(positions[c]) - vec3(positions[a]));
        sums[a] += n;
        sums[b] += n;
        sums[c] += n;
    }
    sums.into_iter().map(|n| array3(normalized_or(n, Vec3::Y))).collect()
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

/// Mirror geometry across the plane perpendicular to `axis` (0=X, 1=Y, 2=Z)
/// at coordinate zero.
///
/// The result is the original mesh followed by a mirrored copy: positions and
/// normals have their `axis` component negated, UVs are preserved, and the
/// mirrored copy has reversed triangle winding so it stays outward-facing.
pub fn mirror(mesh: &mut MeshData, axis: usize) -> Result<(), String> {
    if axis > 2 {
        return Err(format!("Mirror axis {axis} is out of range (expected 0, 1 or 2)"));
    }
    if mesh.positions.is_empty() {
        return Err("Cannot mirror an empty mesh".to_string());
    }

    let base_v = mesh.positions.len() as u32;
    let normals = normals_of(mesh);
    let uvs = uvs_of(mesh);
    let old_positions = mesh.positions.clone();
    let old_indices = mesh.indices.clone();

    let mut positions = Vec::with_capacity(old_positions.len() * 2);
    let mut out_normals = Vec::with_capacity(old_positions.len() * 2);
    let mut out_uvs = Vec::with_capacity(old_positions.len() * 2);
    let mut indices = Vec::with_capacity(old_indices.len() * 2);

    // Original half, untouched.
    positions.extend_from_slice(&old_positions);
    out_normals.extend_from_slice(&normals);
    out_uvs.extend_from_slice(&uvs);
    indices.extend_from_slice(&old_indices);

    // Mirrored half: negate the axis component of positions and normals.
    for i in 0..old_positions.len() {
        let mut p = old_positions[i];
        p[axis] = -p[axis];
        positions.push(p);

        let mut n = normals[i];
        n[axis] = -n[axis];
        out_normals.push(n);

        out_uvs.push(uvs[i]);
    }
    for tri in old_indices.chunks(3) {
        if tri.len() == 3 {
            indices.extend_from_slice(&[tri[0] + base_v, tri[2] + base_v, tri[1] + base_v]);
        } else {
            indices.extend(tri.iter().map(|&i| i + base_v));
        }
    }

    *mesh = MeshData {
        positions,
        normals: out_normals,
        uvs: out_uvs,
        indices,
    };
    Ok(())
}

/// Return a new mesh holding `count` copies of `mesh`, each translated by
/// `offset * i`. Normals and UVs are preserved and indices are offset into the
/// correct copy. `count` is clamped to `1..=64`; empty meshes yield an empty
/// result.
pub fn array(mesh: &MeshData, count: u32, offset: [f32; 3]) -> MeshData {
    let count = count.clamp(1, 64) as usize;
    let mut out = MeshData::empty();
    if mesh.positions.is_empty() {
        return out;
    }

    let base_v = mesh.positions.len();
    let normals = normals_of(mesh);
    let uvs = uvs_of(mesh);
    out.positions.reserve(base_v * count);
    out.normals.reserve(base_v * count);
    out.uvs.reserve(base_v * count);
    out.indices.reserve(mesh.indices.len() * count);

    let step = vec3(offset);
    for i in 0..count {
        let base = (i * base_v) as u32;
        let shift = step * i as f32;
        for &p in &mesh.positions {
            out.positions.push(array3(vec3(p) + shift));
        }
        out.normals.extend_from_slice(&normals);
        out.uvs.extend_from_slice(&uvs);
        out.indices.extend(mesh.indices.iter().map(|&idx| idx + base));
    }
    out
}

/// Build a closed thin shell: keep the original outer shell, add an inner
/// shell offset along `-normal` by `thickness` with reversed winding, and
/// stitch boundary edges with quads (two triangles each).
///
/// Boundary walls get their own four vertices so their normal stays flat,
/// matching [`extrude`]. Returns an error for an empty mesh or zero/non-finite
/// thickness.
pub fn solidify(mesh: &mut MeshData, thickness: f32) -> Result<(), String> {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        return Err("Cannot solidify an empty mesh".to_string());
    }
    if thickness == 0.0 || !thickness.is_finite() {
        return Err("Solidify thickness must be a non-zero, finite value".to_string());
    }

    let base_v = mesh.positions.len() as u32;
    let normals = normals_of(mesh);
    let uvs = uvs_of(mesh);
    let old_positions = mesh.positions.clone();
    let old_indices = mesh.indices.clone();

    let mut positions = Vec::with_capacity(old_positions.len() * 2);
    let mut out_normals = Vec::with_capacity(old_positions.len() * 2);
    let mut out_uvs = Vec::with_capacity(old_positions.len() * 2);
    let mut indices = Vec::with_capacity(old_indices.len() * 2);

    // Outer shell, unchanged.
    positions.extend_from_slice(&old_positions);
    out_normals.extend_from_slice(&normals);
    out_uvs.extend_from_slice(&uvs);
    indices.extend_from_slice(&old_indices);

    // Inner shell: offset along -normal, reversed winding.
    for (i, &p) in old_positions.iter().enumerate() {
        let inward = normalized_or(vec3(normals[i]), Vec3::Y) * thickness;
        positions.push(array3(vec3(p) - inward));
        out_normals.push(normals[i]);
        out_uvs.push(uvs[i]);
    }
    for tri in old_indices.chunks(3) {
        if tri.len() == 3 {
            indices.extend_from_slice(&[tri[0] + base_v, tri[2] + base_v, tri[1] + base_v]);
        } else {
            indices.extend(tri.iter().map(|&i| i + base_v));
        }
    }

    // Undirected edges used by exactly one triangle are boundary edges. Keep
    // the directed form from that triangle so the wall winding closes cleanly.
    struct BoundaryEdge {
        a: u32,
        b: u32,
        count: usize,
    }
    let mut edges: HashMap<(u32, u32), BoundaryEdge> = HashMap::new();
    for tri in old_indices.chunks_exact(3) {
        let ids = [tri[0], tri[1], tri[2]];
        for &(u, v) in &[(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
            let key = if u <= v { (u, v) } else { (v, u) };
            let entry = edges
                .entry(key)
                .or_insert(BoundaryEdge { a: u, b: v, count: 0 });
            entry.count += 1;
        }
    }
    let mut boundary: Vec<BoundaryEdge> =
        edges.into_values().filter(|e| e.count == 1).collect();
    boundary.sort_by_key(|e| (e.a, e.b));

    for e in boundary {
        let pa = vec3(old_positions[e.a as usize]);
        let pb = vec3(old_positions[e.b as usize]);
        let ia = vec3(positions[base_v as usize + e.a as usize]);
        let ib = vec3(positions[base_v as usize + e.b as usize]);
        let inward = (ia - pa + (ib - pb)) * 0.5;
        let wall_normal = normalized_or(inward.cross(pb - pa), Vec3::Y);

        let start = positions.len() as u32;
        // Order: outer-a, inner-a, inner-b, outer-b.
        positions.extend_from_slice(&[array3(pa), array3(ia), array3(ib), array3(pb)]);
        out_normals.extend_from_slice(&[array3(wall_normal); 4]);
        out_uvs.extend_from_slice(&[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]);
        indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    }

    *mesh = MeshData {
        positions,
        normals: out_normals,
        uvs: out_uvs,
        indices,
    };
    Ok(())
}

/// Laplacian smoothing: repeatedly move every vertex toward the average of
/// its (unique) connected neighbours by `factor`. `factor` is clamped to
/// `0.0..=1.0` and `iterations` to `0..=10`. Normals are recalculated at the
/// end. Returns an error for an empty mesh.
pub fn smooth(mesh: &mut MeshData, factor: f32, iterations: u32) -> Result<(), String> {
    if mesh.positions.is_empty() {
        return Err("Cannot smooth an empty mesh".to_string());
    }

    let factor = if factor.is_finite() {
        factor.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let iterations = iterations.min(10);

    if iterations > 0 && factor > 0.0 {
        let vertex_count = mesh.positions.len();
        for _ in 0..iterations {
            let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); vertex_count];
            for tri in mesh.indices.chunks_exact(3) {
                let ids = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
                if ids.iter().any(|&i| i >= vertex_count) {
                    continue;
                }
                for &(a, b) in &[(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
                    neighbours[a].push(b);
                    neighbours[b].push(a);
                }
            }
            for ns in &mut neighbours {
                ns.sort_unstable();
                ns.dedup();
            }

            let mut new_positions = mesh.positions.clone();
            for (i, &p) in mesh.positions.iter().enumerate() {
                let ns = &neighbours[i];
                if ns.is_empty() {
                    continue;
                }
                let mut sum = Vec3::ZERO;
                for &j in ns {
                    sum += vec3(mesh.positions[j]);
                }
                let average = sum / ns.len() as f32;
                let current = vec3(p);
                new_positions[i] = array3(current + (average - current) * factor);
            }
            mesh.positions = new_positions;
        }
    }

    mesh.recalculate_normals();
    Ok(())
}

/// Validation / normalisation pass: copy `indices` as triangle chunks. Returns
/// an error when the index buffer ends in a chunk with fewer than three
/// entries.
pub fn triangulate(mesh: &mut MeshData) -> Result<(), String> {
    let mut indices = Vec::with_capacity(mesh.indices.len());
    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            return Err(format!("Trailing {} index/indices do not form a triangle", tri.len()));
        }
        indices.extend_from_slice(&[tri[0], tri[1], tri[2]]);
    }
    mesh.indices = indices;
    Ok(())
}

/// Build a [`MeshData`] from explicit vertex positions and arbitrary polygon
/// faces (three or more indices each), the equivalent of Blender's
/// `bmesh.from_pydata`.
///
/// Polygons are triangulated as a fan around their first index, so an n-gon
/// yields `n - 2` triangles whose winding matches the input face order.
/// Smooth vertex normals are generated by [`compute_smooth_normals`] and UVs
/// by [`box_project_uvs`] with `scale = 1.0`.
///
/// Returns an error for empty input, a face with fewer than three indices, or
/// a face index outside `vertices`.
pub fn from_faces(vertices: &[[f32; 3]], faces: &[Vec<usize>]) -> Result<MeshData, String> {
    if vertices.is_empty() || faces.is_empty() {
        return Err("Cannot build a mesh from empty vertices or faces".to_string());
    }
    for (fi, face) in faces.iter().enumerate() {
        if face.len() < 3 {
            return Err(format!("Face {fi} has fewer than 3 indices"));
        }
        for &i in face {
            if i >= vertices.len() {
                return Err(format!(
                    "Face {fi} references vertex {i} out of range ({} vertices)",
                    vertices.len()
                ));
            }
        }
    }

    let mut indices = Vec::new();
    for face in faces {
        for k in 1..face.len() - 1 {
            indices.extend_from_slice(&[face[0] as u32, face[k] as u32, face[k + 1] as u32]);
        }
    }

    let normals = compute_smooth_normals(vertices, &indices);
    let mut mesh = MeshData {
        positions: vertices.to_vec(),
        normals,
        uvs: vec![[0.0, 0.0]; vertices.len()],
        indices,
    };
    box_project_uvs(&mut mesh, 1.0);
    Ok(mesh)
}

/// Extrude a set of TRIANGLES, identified by their position in the index
/// buffer (`indices[3 * f .. 3 * f + 3]`).
///
/// Each selected triangle gets three duplicated vertices pushed along its
/// averaged vertex normal (or its geometric face normal when normals are
/// missing) by `distance`, a new top triangle, and three side quads of two
/// triangles each with their own flat-shaded vertices. The original selected
/// triangles are removed; unselected ones and all original vertices are kept.
///
/// Returns an error for an empty mesh, a malformed index buffer, an
/// out-of-range face id, or a zero / non-finite distance.
pub fn extrude_faces(mesh: &mut MeshData, faces: &[usize], distance: f32) -> Result<(), String> {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        return Err("Cannot extrude faces of an empty mesh".to_string());
    }
    if mesh.indices.len() % 3 != 0 {
        return Err("Index buffer length is not a multiple of 3".to_string());
    }
    let face_count = mesh.indices.len() / 3;
    let mut selected = vec![false; face_count];
    for &f in faces {
        if f >= face_count {
            return Err(format!(
                "Face index {f} out of range (mesh has {face_count} triangles)"
            ));
        }
        selected[f] = true;
    }
    if distance == 0.0 || !distance.is_finite() {
        return Err("Extrusion distance must be a non-zero, finite value".to_string());
    }
    for (i, &idx) in mesh.indices.iter().enumerate() {
        if idx as usize >= mesh.positions.len() {
            return Err(format!(
                "Index {idx} at {i} is out of range for {} vertices",
                mesh.positions.len()
            ));
        }
    }

    let normals = normals_of(mesh);
    let uvs = uvs_of(mesh);
    let have_normals = mesh.normals.len() == mesh.positions.len();

    let mut positions = mesh.positions.clone();
    let mut out_normals = normals.clone();
    let mut out_uvs = uvs.clone();
    // Every selected face contributes 1 top + 3 * 2 side triangles.
    let mut indices = Vec::with_capacity(mesh.indices.len() + faces.len() * 9);

    // Keep every triangle that was not selected.
    for (f, tri) in mesh.indices.chunks_exact(3).enumerate() {
        if !selected[f] {
            indices.extend_from_slice(tri);
        }
    }

    for (f, tri) in mesh.indices.chunks_exact(3).enumerate() {
        if !selected[f] {
            continue;
        }
        let ids = [tri[0], tri[1], tri[2]];
        let a = ids[0] as usize;
        let b = ids[1] as usize;
        let c = ids[2] as usize;
        let face_n = face_normal(&mesh.positions, &ids);
        let normal = if have_normals {
            normalized_or(vec3(normals[a]) + vec3(normals[b]) + vec3(normals[c]), face_n)
        } else {
            face_n
        };
        let offset = normal * distance;
        let flip = face_n.dot(offset) < 0.0;

        // Top triangle: three fresh vertices carrying the averaged normal.
        let top = positions.len() as u32;
        for &id in &ids {
            positions.push(array3(vec3(mesh.positions[id as usize]) + offset));
            out_normals.push(array3(normal));
            out_uvs.push(uvs[id as usize]);
        }
        indices.extend_from_slice(&[top, top + 1, top + 2]);

        // Three side quads, each with four fresh vertices so its normal stays
        // flat (matching the behaviour of `extrude`).
        for e in 0..3 {
            let ia = ids[e] as usize;
            let ib = ids[(e + 1) % 3] as usize;
            let pa = vec3(mesh.positions[ia]);
            let pb = vec3(mesh.positions[ib]);
            let mut side = normalized_or((pb - pa).cross(offset), Vec3::Y);
            if flip {
                side = -side;
            }
            let flat = array3(side);
            let len = (pb - pa).length();
            let height = distance.abs();

            let start = positions.len() as u32;
            positions.extend_from_slice(&[
                array3(pa),
                array3(pb),
                array3(pa + offset),
                array3(pb + offset),
            ]);
            out_normals.extend_from_slice(&[flat; 4]);
            out_uvs.extend_from_slice(&[[0.0, 0.0], [len, 0.0], [0.0, height], [len, height]]);

            if flip {
                indices.extend_from_slice(&[start, start + 3, start + 1, start, start + 2, start + 3]);
            } else {
                indices.extend_from_slice(&[start, start + 1, start + 3, start, start + 3, start + 2]);
            }
        }
    }

    *mesh = MeshData { positions, normals: out_normals, uvs: out_uvs, indices };
    Ok(())
}

/// Inset a set of TRIANGLES (identified by their index-buffer chunk id).
///
/// Each selected triangle is replaced by a smaller inner triangle and three
/// border quads (two triangles each). Its three corners move toward the
/// centroid by at most `0.45 * shortest_edge` to keep the inner triangle from
/// inverting. Inner vertices reuse the original normals/UVs (lerped toward
/// the centroid UV); the border quads get fresh vertices with flat normals.
///
/// Returns an error for an empty mesh, a malformed index buffer, an
/// out-of-range face id, or a non-positive / non-finite amount.
pub fn inset_faces(mesh: &mut MeshData, faces: &[usize], amount: f32) -> Result<(), String> {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        return Err("Cannot inset faces of an empty mesh".to_string());
    }
    if mesh.indices.len() % 3 != 0 {
        return Err("Index buffer length is not a multiple of 3".to_string());
    }
    let face_count = mesh.indices.len() / 3;
    let mut selected = vec![false; face_count];
    for &f in faces {
        if f >= face_count {
            return Err(format!(
                "Face index {f} out of range (mesh has {face_count} triangles)"
            ));
        }
        selected[f] = true;
    }
    if amount <= 0.0 || !amount.is_finite() {
        return Err("Inset amount must be a positive, finite value".to_string());
    }
    for (i, &idx) in mesh.indices.iter().enumerate() {
        if idx as usize >= mesh.positions.len() {
            return Err(format!(
                "Index {idx} at {i} is out of range for {} vertices",
                mesh.positions.len()
            ));
        }
    }

    let normals = normals_of(mesh);
    let uvs = uvs_of(mesh);
    let have_normals = mesh.normals.len() == mesh.positions.len();

    let mut positions = mesh.positions.clone();
    let mut out_normals = normals.clone();
    let mut out_uvs = uvs.clone();
    let mut indices = Vec::with_capacity(mesh.indices.len() + faces.len() * 9);

    for (f, tri) in mesh.indices.chunks_exact(3).enumerate() {
        if !selected[f] {
            indices.extend_from_slice(tri);
        }
    }

    for (f, tri) in mesh.indices.chunks_exact(3).enumerate() {
        if !selected[f] {
            continue;
        }
        let ids = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let face_n = face_normal(&mesh.positions, &[tri[0], tri[1], tri[2]]);
        let pa = vec3(mesh.positions[ids[0]]);
        let pb = vec3(mesh.positions[ids[1]]);
        let pc = vec3(mesh.positions[ids[2]]);
        let centroid = (pa + pb + pc) / 3.0;
        let uv_centroid = [
            (uvs[ids[0]][0] + uvs[ids[1]][0] + uvs[ids[2]][0]) / 3.0,
            (uvs[ids[0]][1] + uvs[ids[1]][1] + uvs[ids[2]][1]) / 3.0,
        ];
        let min_edge = (pb - pa)
            .length()
            .min((pc - pb).length())
            .min((pa - pc).length());
        let shift = amount.min(0.45 * min_edge);

        // Three inner vertices, moved toward the centroid.
        let inner = positions.len() as u32;
        for &id in &ids {
            let p = vec3(mesh.positions[id]);
            let toward = centroid - p;
            let dist = toward.length();
            let moved = if dist > 1e-6 { p + toward * (shift / dist) } else { p };
            positions.push(array3(moved));
            let n = if have_normals { vec3(normals[id]) } else { face_n };
            out_normals.push(array3(n));
            let frac = if dist > 1e-6 { shift / dist } else { 0.0 };
            out_uvs.push([
                uvs[id][0] + (uv_centroid[0] - uvs[id][0]) * frac,
                uvs[id][1] + (uv_centroid[1] - uvs[id][1]) * frac,
            ]);
        }
        indices.extend_from_slice(&[inner, inner + 1, inner + 2]);

        // Three border quads between each original edge and its inner edge.
        for e in 0..3 {
            let ia = ids[e];
            let ib = ids[(e + 1) % 3];
            let in_a = inner as usize + e;
            let in_b = inner as usize + (e + 1) % 3;
            let pia = vec3(positions[in_a]);
            let pib = vec3(positions[in_b]);
            let flat = array3(normalized_or(
                (vec3(mesh.positions[ib]) - vec3(mesh.positions[ia]))
                    .cross(pib - pia),
                face_n,
            ));
            let uv_in_a = out_uvs[in_a];
            let uv_in_b = out_uvs[in_b];

            let start = positions.len() as u32;
            positions.extend_from_slice(&[
                mesh.positions[ia],
                mesh.positions[ib],
                array3(pib),
                array3(pia),
            ]);
            out_normals.extend_from_slice(&[flat; 4]);
            out_uvs.extend_from_slice(&[uvs[ia], uvs[ib], uv_in_b, uv_in_a]);
            indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
        }
    }

    *mesh = MeshData { positions, normals: out_normals, uvs: out_uvs, indices };
    Ok(())
}

/// Weld vertices closer than `epsilon` into single vertices.
///
/// Vertices are clustered greedily in index order against the first member of
/// each cluster; merged vertices get an averaged position, normal and UV, and
/// every index is remapped to its cluster. Smooth normals are recomputed over
/// the welded topology at the end.
///
/// Returns an error for an empty mesh or a non-positive / non-finite epsilon.
pub fn merge_by_distance(mesh: &mut MeshData, epsilon: f32) -> Result<(), String> {
    if mesh.positions.is_empty() {
        return Err("Cannot merge vertices of an empty mesh".to_string());
    }
    if epsilon <= 0.0 || !epsilon.is_finite() {
        return Err("Merge epsilon must be a positive, finite value".to_string());
    }
    for (i, &idx) in mesh.indices.iter().enumerate() {
        if idx as usize >= mesh.positions.len() {
            return Err(format!(
                "Index {idx} at {i} is out of range for {} vertices",
                mesh.positions.len()
            ));
        }
    }

    let positions = mesh.positions.clone();
    let normals = normals_of(mesh);
    let uvs = uvs_of(mesh);
    let n = positions.len();

    let mut cluster_of = vec![0usize; n];
    let mut members: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let p = vec3(positions[i]);
        let mut found = None;
        for (ci, group) in members.iter().enumerate() {
            let r = group[0];
            if (p - vec3(positions[r])).length() < epsilon {
                found = Some(ci);
                break;
            }
        }
        let ci = match found {
            Some(ci) => ci,
            None => {
                members.push(Vec::new());
                members.len() - 1
            }
        };
        cluster_of[i] = ci;
        members[ci].push(i);
    }

    let mut new_positions = Vec::with_capacity(members.len());
    let mut new_uvs = Vec::with_capacity(members.len());
    for group in &members {
        let inv = 1.0 / group.len() as f32;
        let mut psum = Vec3::ZERO;
        let mut nsum = Vec3::ZERO;
        let mut usum = [0.0f32; 2];
        for &m in group {
            psum += vec3(positions[m]);
            nsum += vec3(normals[m]);
            usum[0] += uvs[m][0];
            usum[1] += uvs[m][1];
        }
        let representative = group[0];
        new_positions.push(array3(psum * inv));
        // Kept for clarity: the averaged normal is computed here, then the
        // final pass below recomputes smooth normals for the welded topology.
        let _averaged_normal = array3(normalized_or(nsum * inv, vec3(normals[representative])));
        new_uvs.push([usum[0] * inv, usum[1] * inv]);
    }

    let new_indices: Vec<u32> = mesh
        .indices
        .iter()
        .map(|&i| cluster_of[i as usize] as u32)
        .collect();
    let new_normals = compute_smooth_normals(&new_positions, &new_indices);

    *mesh = MeshData { positions: new_positions, normals: new_normals, uvs: new_uvs, indices: new_indices };
    Ok(())
}

/// Reverse the winding of every triangle and negate every stored normal.
///
/// Returns an error for an empty mesh.
pub fn flip_normals(mesh: &mut MeshData) -> Result<(), String> {
    if mesh.positions.is_empty() {
        return Err("Cannot flip normals of an empty mesh".to_string());
    }
    for tri in mesh.indices.chunks_exact_mut(3) {
        tri.swap(1, 2);
    }
    for n in &mut mesh.normals {
        n[0] = -n[0];
        n[1] = -n[1];
        n[2] = -n[2];
    }
    Ok(())
}

/// Approximate edge "chamfer".
///
/// Each vertex is nudged toward the average of its unique edge-neighbours by
/// `amount` (clamped to `0.0..=0.5`), then smooth normals are recomputed.
///
/// **This is an approximation, not a true edge bevel.** A real bevel inserts
/// new strips of geometry along each edge with a controllable width and
/// segment count; this operation only relaxes the existing vertices, which
/// rounds/chamfers the silhouette but does not create dedicated bevel faces.
///
/// Returns an error for an empty mesh.
pub fn bevel(mesh: &mut MeshData, amount: f32) -> Result<(), String> {
    if mesh.positions.is_empty() {
        return Err("Cannot bevel an empty mesh".to_string());
    }
    let amount = if amount.is_finite() { amount.clamp(0.0, 0.5) } else { 0.0 };

    let vertex_count = mesh.positions.len();
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); vertex_count];
    for tri in mesh.indices.chunks_exact(3) {
        let ids = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        if ids.iter().any(|&i| i >= vertex_count) {
            continue;
        }
        for &(a, b) in &[(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
            neighbours[a].push(b);
            neighbours[b].push(a);
        }
    }
    for ns in &mut neighbours {
        ns.sort_unstable();
        ns.dedup();
    }

    if amount > 0.0 {
        let mut new_positions = mesh.positions.clone();
        for (i, &p) in mesh.positions.iter().enumerate() {
            let ns = &neighbours[i];
            if ns.is_empty() {
                continue;
            }
            let mut sum = Vec3::ZERO;
            for &j in ns {
                sum += vec3(mesh.positions[j]);
            }
            let avg = sum / ns.len() as f32;
            let current = vec3(p);
            new_positions[i] = array3(current + (avg - current) * amount);
        }
        mesh.positions = new_positions;
    }

    mesh.normals = compute_smooth_normals(&mesh.positions, &mesh.indices);
    Ok(())
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

    fn asymmetric_triangle() -> MeshData {
        MeshData {
            positions: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            normals: vec![[1.0, 2.0, 3.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
        }
    }

    #[test]
    fn mirror_doubles_geometry_and_negates_axis() {
        for axis in 0..3 {
            let mut mesh = asymmetric_triangle();
            mirror(&mut mesh, axis).unwrap();

            assert_eq!(mesh.positions.len(), 6);
            assert_eq!(mesh.normals.len(), 6);
            assert_eq!(mesh.uvs.len(), 6);
            assert_eq!(mesh.indices.len(), 6);

            for i in 0..3 {
                for c in 0..3 {
                    let expected_p = if c == axis {
                        -mesh.positions[i][c]
                    } else {
                        mesh.positions[i][c]
                    };
                    let expected_n = if c == axis {
                        -mesh.normals[i][c]
                    } else {
                        mesh.normals[i][c]
                    };
                    assert!((mesh.positions[i + 3][c] - expected_p).abs() < 1e-6);
                    assert!((mesh.normals[i + 3][c] - expected_n).abs() < 1e-6);
                }
                assert_eq!(mesh.uvs[i + 3], mesh.uvs[i]);
            }

            // Original winding kept, mirrored winding reversed.
            assert_eq!(&mesh.indices[..3], &[0, 1, 2]);
            assert_eq!(&mesh.indices[3..], &[3, 5, 4]);
        }
    }

    #[test]
    fn mirror_rejects_invalid_axis_and_empty() {
        assert!(mirror(&mut triangle(), 3).is_err());
        assert!(mirror(&mut triangle(), usize::MAX).is_err());
        assert!(mirror(&mut MeshData::empty(), 0).is_err());
    }

    #[test]
    fn array_copies_translate_and_offset_indices() {
        let mesh = triangle();
        let out = array(&mesh, 3, [1.0, 2.0, 3.0]);

        assert_eq!(out.positions.len(), 9);
        assert_eq!(out.normals.len(), 9);
        assert_eq!(out.uvs.len(), 9);
        assert_eq!(out.indices.len(), 9);
        assert_eq!(&out.indices[0..3], &[0, 1, 2]);
        assert_eq!(&out.indices[3..6], &[3, 4, 5]);
        assert_eq!(&out.indices[6..9], &[6, 7, 8]);

        // Copy 0 is unchanged, copy i is translated by offset * i.
        assert_eq!(out.positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(out.positions[3], [1.0, 2.0, 3.0]);
        assert_eq!(out.positions[6], [2.0, 4.0, 6.0]);
        assert_eq!(out.normals[4], [0.0, 0.0, 1.0]);
        assert_eq!(out.uvs[4], [1.0, 0.0]);
    }

    #[test]
    fn array_clamps_count_and_handles_empty() {
        let mesh = triangle();
        assert_eq!(array(&mesh, 0, [0.0; 3]).positions.len(), 3);
        assert_eq!(array(&mesh, 100, [0.0; 3]).positions.len(), 3 * 64);
        assert_eq!(array(&mesh, 100, [0.0; 3]).indices.len(), 3 * 64);

        let empty = array(&MeshData::empty(), 4, [1.0, 1.0, 1.0]);
        assert!(empty.positions.is_empty());
        assert!(empty.normals.is_empty());
        assert!(empty.uvs.is_empty());
        assert!(empty.indices.is_empty());
    }

    #[test]
    fn solidify_triangle_counts() {
        let mut mesh = triangle();
        solidify(&mut mesh, 0.1).unwrap();

        // 3 outer + 3 inner + 3 boundary edges * 4 wall vertices = 18.
        assert_eq!(mesh.positions.len(), 18);
        // 2 shells * 1 triangle + 3 walls * 2 triangles = 8 triangles.
        assert_eq!(mesh.indices.len(), 24);
        assert_eq!(mesh.normals.len(), 18);
        assert_eq!(mesh.uvs.len(), 18);

        // Inner shell is displaced along -normal (the triangle normal is +Z).
        assert!((mesh.positions[3][2] + 0.1).abs() < 1e-6);
        // Inner winding is the reverse of the outer winding.
        assert_eq!(&mesh.indices[3..6], &[3, 5, 4]);
    }

    #[test]
    fn solidify_rejects_invalid_input() {
        assert!(solidify(&mut MeshData::empty(), 0.1).is_err());
        assert!(solidify(&mut triangle(), 0.0).is_err());
        assert!(solidify(&mut triangle(), f32::NAN).is_err());
    }

    #[test]
    fn smooth_moves_toward_neighbour_average() {
        let mut mesh = MeshData {
            positions: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0]; 3],
            indices: vec![0, 1, 2],
        };
        smooth(&mut mesh, 1.0, 1).unwrap();

        // Vertex 1 has neighbours 0 and 2; their average is (0.5, 1, 0).
        assert!((mesh.positions[1][0] - 0.5).abs() < 1e-6);
        assert!((mesh.positions[1][1] - 1.0).abs() < 1e-6);
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.normals.len(), 3);
    }

    #[test]
    fn smooth_clamps_factor_and_recalculates_normals() {
        let base = MeshData {
            positions: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0]; 3],
            indices: vec![0, 1, 2],
        };
        let mut clamped = base.clone();
        let mut full = base.clone();
        smooth(&mut clamped, 5.0, 1).unwrap();
        smooth(&mut full, 1.0, 1).unwrap();
        assert_eq!(clamped.positions, full.positions);

        // Zero iterations is still a normal-recalculation pass.
        let mut untouched = base;
        untouched.normals.clear();
        smooth(&mut untouched, 0.5, 0).unwrap();
        assert_eq!(untouched.normals.len(), 3);
        assert_eq!(untouched.positions[1], [2.0, 0.0, 0.0]);

        assert!(smooth(&mut MeshData::empty(), 1.0, 1).is_err());
    }

    #[test]
    fn triangulate_copies_chunks_and_rejects_partial_chunk() {
        let mut mesh = quad();
        triangulate(&mut mesh).unwrap();
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);

        let mut bad = quad();
        bad.indices.push(0);
        assert!(triangulate(&mut bad).is_err());

        let mut empty = MeshData::empty();
        assert!(triangulate(&mut empty).is_ok());
        assert!(empty.indices.is_empty());
    }
}

#[cfg(test)]
mod mesh_ops_extra_tests {
    use super::*;

    /// A single CCW triangle in the XY plane, normal +Z.
    fn tri() -> MeshData {
        MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
        }
    }

    #[test]
    fn from_faces_builds_quad_as_four_verts_two_triangles() {
        let mesh = from_faces(
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            &[vec![0, 1, 2, 3]],
        )
        .unwrap();

        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(mesh.normals.len(), 4);
        assert_eq!(mesh.uvs.len(), 4);
        assert!((mesh.normals[0][2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn from_faces_rejects_invalid_input() {
        assert!(from_faces(&[], &[vec![0, 1, 2]]).is_err());
        assert!(from_faces(&[[0.0; 3]; 3], &[]).is_err());
        assert!(from_faces(&[[0.0; 3]; 3], &[vec![0, 1]]).is_err());
        assert!(from_faces(&[[0.0; 3]; 3], &[vec![0, 1, 3]]).is_err());
    }

    #[test]
    fn extrude_faces_grows_vertex_and_triangle_counts() {
        let mut mesh = tri();
        extrude_faces(&mut mesh, &[0], 1.0).unwrap();

        // 3 original + 3 top + 3 walls * 4 = 18 vertices.
        assert_eq!(mesh.positions.len(), 18);
        // 1 top + 3 walls * 2 = 7 triangles.
        assert_eq!(mesh.indices.len(), 21);
        assert_eq!(mesh.normals.len(), 18);
        assert_eq!(mesh.uvs.len(), 18);
        // The original triangle was removed; the top copy uses vertices 3..6.
        assert_eq!(&mesh.indices[..3], &[3, 4, 5]);
        assert!((mesh.positions[3][2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn extrude_faces_rejects_invalid_input() {
        assert!(extrude_faces(&mut MeshData::empty(), &[0], 1.0).is_err());
        let mut mesh = tri();
        assert!(extrude_faces(&mut mesh, &[1], 1.0).is_err());
        assert!(extrude_faces(&mut mesh, &[0], 0.0).is_err());
    }

    #[test]
    fn inset_faces_creates_inner_triangle_and_border_quads() {
        let mut mesh = tri();
        inset_faces(&mut mesh, &[0], 0.2).unwrap();

        // 3 original + 3 inner + 3 borders * 4 = 18 vertices.
        assert_eq!(mesh.positions.len(), 18);
        // 1 inner + 3 borders * 2 = 7 triangles.
        assert_eq!(mesh.indices.len(), 21);
        assert_eq!(mesh.normals.len(), 18);
        assert_eq!(mesh.uvs.len(), 18);
        assert_eq!(&mesh.indices[..3], &[3, 4, 5]);

        // Each inner vertex moved strictly inside the original triangle.
        for &i in &[3usize, 4, 5] {
            let p = mesh.positions[i];
            assert!(p[0] > 0.0 && p[1] > 0.0 && p[0] + p[1] < 1.0);
        }
    }

    #[test]
    fn inset_faces_rejects_invalid_input() {
        assert!(inset_faces(&mut MeshData::empty(), &[0], 0.1).is_err());
        let mut mesh = tri();
        assert!(inset_faces(&mut mesh, &[1], 0.1).is_err());
        assert!(inset_faces(&mut mesh, &[0], 0.0).is_err());
        assert!(inset_faces(&mut mesh, &[0], -1.0).is_err());
    }

    #[test]
    fn merge_by_distance_welds_duplicates_and_remaps_indices() {
        let mut mesh = MeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1e-4, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 3, 1, 2, 3],
        };
        merge_by_distance(&mut mesh, 1e-3).unwrap();

        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.normals.len(), 3);
        assert_eq!(mesh.uvs.len(), 3);
        // Old 0 and 1 merged to cluster 0, 2 -> 1, 3 -> 2.
        assert_eq!(mesh.indices, vec![0, 0, 2, 0, 1, 2]);
        assert!((mesh.uvs[0][0] - 0.05).abs() < 1e-5);
    }

    #[test]
    fn merge_by_distance_rejects_invalid_input() {
        assert!(merge_by_distance(&mut MeshData::empty(), 0.01).is_err());
        let mut mesh = tri();
        assert!(merge_by_distance(&mut mesh, 0.0).is_err());
        assert!(merge_by_distance(&mut mesh, -1.0).is_err());
    }

    #[test]
    fn flip_normals_reverses_winding_and_negates_normals() {
        let mut mesh = tri();
        flip_normals(&mut mesh).unwrap();

        assert_eq!(mesh.indices, vec![0, 2, 1]);
        assert_eq!(mesh.normals, vec![[0.0, 0.0, -1.0]; 3]);
        assert!(flip_normals(&mut MeshData::empty()).is_err());
    }

    #[test]
    fn bevel_moves_vertex_toward_neighbours_and_recomputes_normals() {
        let mut mesh = MeshData {
            positions: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            normals: vec![[0.0, 1.0, 0.0]; 3], // deliberately wrong
            uvs: vec![[0.0, 0.0]; 3],
            indices: vec![0, 1, 2],
        };
        bevel(&mut mesh, 0.5).unwrap();

        // Neighbours of vertex 0 average to (1, 1, 0); move halfway there.
        assert!((mesh.positions[0][0] - 0.5).abs() < 1e-6);
        assert!((mesh.positions[0][1] - 0.5).abs() < 1e-6);
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.normals.len(), 3);
        assert!((mesh.normals[0][2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bevel_clamps_amount_and_rejects_empty() {
        let base = MeshData {
            positions: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0]; 3],
            indices: vec![0, 1, 2],
        };
        let mut clamped = base.clone();
        let mut half = base;
        bevel(&mut clamped, 9.0).unwrap();
        bevel(&mut half, 0.5).unwrap();
        assert_eq!(clamped.positions, half.positions);
        assert!(bevel(&mut MeshData::empty(), 0.5).is_err());
    }
}
