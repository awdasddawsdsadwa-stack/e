//! Texture loading, downscaling and procedural pattern generation.
//!
//! All textures are kept as CPU-side RGBA8 buffers matching
//! [`crate::scene::TextureData`] (row-major, top-left origin) and are clamped
//! to [`MAX_TEXTURE_SIZE`] before they ever reach the GPU.

use crate::scene::TextureData;
use image::imageops::{resize, FilterType};
use image::RgbaImage;

/// Largest width/height allowed for any single texture.
pub const MAX_TEXTURE_SIZE: u32 = 1024;

const TAU: f32 = std::f32::consts::TAU;

// ---------------------------------------------------------------------------
// Loading / resizing
// ---------------------------------------------------------------------------

/// Shrink `tex` in place, preserving aspect ratio, when either dimension
/// exceeds `max`. Uses a triangle (bilinear) filter via the `image` crate.
/// Pixel data stays row-major RGBA8. A `max` of 0 is treated as 1 so the
/// texture never collapses to zero-sized dimensions.
pub fn downscale(tex: &mut TextureData, max: u32) {
    let max = max.max(1);
    let longest = tex.width.max(tex.height);
    if longest <= max || tex.width == 0 || tex.height == 0 {
        return;
    }

    let scale = max as f32 / longest as f32;
    let new_w = ((tex.width as f32 * scale).floor() as u32).max(1);
    let new_h = ((tex.height as f32 * scale).floor() as u32).max(1);
    if new_w == tex.width && new_h == tex.height {
        return;
    }

    // Flatten the [u8; 4] rows into the interleaved buffer `image` expects.
    let raw: Vec<u8> = tex.pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let img = match RgbaImage::from_vec(tex.width, tex.height, raw) {
        Some(img) => img,
        None => return, // inconsistent TextureData: leave it untouched
    };

    let resized = resize(&img, new_w, new_h, FilterType::Triangle);
    let pixels: Vec<[u8; 4]> = resized.pixels().map(|p| p.0).collect();

    tex.width = new_w;
    tex.height = new_h;
    tex.pixels = pixels;
}

/// Open a PNG/JPEG from disk, convert it to RGBA8 and clamp it to
/// [`MAX_TEXTURE_SIZE`]. `name` overrides the default name (the file stem).
pub fn from_image_file(path: &str, name: Option<String>) -> Result<TextureData, String> {
    let img = image::open(path)
        .map_err(|e| format!("failed to open image '{}': {}", path, e))?;
    let (width, height) = (img.width(), img.height());
    let rgba = img.into_rgba8();
    let pixels: Vec<[u8; 4]> = rgba.pixels().map(|p| p.0).collect();

    let label = name.unwrap_or_else(|| {
        std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "texture".to_string())
    });

    let mut tex = TextureData { name: label, width, height, pixels };
    downscale(&mut tex, MAX_TEXTURE_SIZE);
    Ok(tex)
}

// ---------------------------------------------------------------------------
// Procedural patterns
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    Checkerboard,
    ValueNoise,
    Wood,
    Brick,
    Gradient,
    Scratches,
}

impl Pattern {
    /// Stable lowercase identifier used in generated texture names.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Checkerboard => "checkerboard",
            Self::ValueNoise => "value_noise",
            Self::Wood => "wood",
            Self::Brick => "brick",
            Self::Gradient => "gradient",
            Self::Scratches => "scratches",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TextureOptions {
    /// First / base colour.
    pub color_a: [u8; 4],
    /// Second / accent colour (mortar for bricks).
    pub color_b: [u8; 4],
    /// Pattern frequency: block size, noise cell count, ring/brick count, ...
    pub scale: f32,
    /// Seed for all noise-based patterns.
    pub seed: u32,
}

impl Default for TextureOptions {
    fn default() -> Self {
        Self {
            color_a: [210, 210, 214, 255],
            color_b: [70, 74, 82, 255],
            scale: 8.0,
            seed: 1337,
        }
    }
}

// --- deterministic noise helpers -------------------------------------------

/// Sanitize a user-supplied scale value (reject NaN, infinities and negatives).
fn clean_scale(s: f32) -> f32 {
    if s.is_finite() && s > 0.0 {
        s
    } else {
        1.0
    }
}

/// Integer hash with good avalanche, used for every lattice value.
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x
}

/// One pseudo-random value in `[0, 1]` for an integer lattice coordinate.
fn lattice(ix: u32, iy: u32, seed: u32) -> f32 {
    let h = hash_u32(
        seed ^ ix.wrapping_mul(0x9E37_79B1) ^ iy.wrapping_mul(0x85EB_CA6B),
    );
    (h & 0xFFFF) as f32 / 65535.0
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Seamless tileable 2D value noise (fractional Brownian motion).
///
/// `u` and `v` are in `[0, 1)`. The lattice wraps per octave, so the value at
/// the right edge equals the value at the left edge (same for top/bottom).
fn tileable_fbm(u: f32, v: f32, cells_x: u32, cells_y: u32, octaves: u32, seed: u32) -> f32 {
    let mut cx = cells_x.max(1);
    let mut cy = cells_y.max(1);
    let mut amplitude = 1.0f32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;

    for octave in 0..octaves.max(1) {
        let fx = u * cx as f32;
        let fy = v * cy as f32;
        let ix = fx.floor() as u32;
        let iy = fy.floor() as u32;
        let tx = smoothstep(fx - fx.floor());
        let ty = smoothstep(fy - fy.floor());

        let x0 = ix % cx;
        let x1 = (ix + 1) % cx;
        let y0 = iy % cy;
        let y1 = (iy + 1) % cy;
        let s = seed.wrapping_add(octave.wrapping_mul(0x9E37_79B1));

        let a = lattice(x0, y0, s);
        let b = lattice(x1, y0, s);
        let c = lattice(x0, y1, s);
        let d = lattice(x1, y1, s);
        let top = a + (b - a) * tx;
        let bottom = c + (d - c) * tx;
        let value = top + (bottom - top) * ty;

        sum += value * amplitude;
        norm += amplitude;
        amplitude *= 0.5;
        cx = cx.saturating_mul(2);
        cy = cy.saturating_mul(2);
    }

    if norm > 0.0 {
        (sum / norm).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Linear RGBA blend, preserving the alpha of `a`.
fn mix(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let mut out = [0u8; 4];
    for i in 0..4 {
        let av = a[i] as f32;
        let bv = if i == 3 { av } else { b[i] as f32 };
        out[i] = (av + (bv - av) * t).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Scale RGB channels by `factor`, leaving alpha untouched.
fn shade(c: [u8; 4], factor: f32) -> [u8; 4] {
    let f = factor.clamp(0.0, 1.0);
    [
        (c[0] as f32 * f).round() as u8,
        (c[1] as f32 * f).round() as u8,
        (c[2] as f32 * f).round() as u8,
        c[3],
    ]
}

// --- generators -------------------------------------------------------------

fn checkerboard(size: u32, o: &TextureOptions) -> Vec<[u8; 4]> {
    let block = (clean_scale(o.scale).round() as u32).max(1);
    let mut pixels = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        for x in 0..size {
            let even = ((x / block) + (y / block)) % 2 == 0;
            pixels.push(if even { o.color_a } else { o.color_b });
        }
    }
    pixels
}

fn value_noise(size: u32, o: &TextureOptions) -> Vec<[u8; 4]> {
    let cells = (clean_scale(o.scale).round() as u32).max(1);
    let mut pixels = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        let v = y as f32 / size as f32;
        for x in 0..size {
            let u = x as f32 / size as f32;
            let n = tileable_fbm(u, v, cells, cells, 4, o.seed);
            pixels.push(mix(o.color_a, o.color_b, n));
        }
    }
    pixels
}

fn wood(size: u32, o: &TextureOptions) -> Vec<[u8; 4]> {
    let s = clean_scale(o.scale);
    let cells = (s * 2.0).round().max(2.0) as u32;
    let rings = (s * 3.0).round().max(1.0) as u32;
    let mut pixels = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        let v = y as f32 / size as f32;
        for x in 0..size {
            let u = x as f32 / size as f32;
            let warp = tileable_fbm(u, v, cells, cells, 3, o.seed);
            let fine = tileable_fbm(u, v, cells.saturating_mul(2), cells.saturating_mul(2), 2, o.seed ^ 0x9E37_79B1);
            let t = u * rings as f32 + (warp - 0.5) * 0.9;
            let ring = 0.5 + 0.5 * (t * TAU).sin();
            let grain = (ring * 0.85 + fine * 0.15).clamp(0.0, 1.0);
            pixels.push(mix(o.color_a, o.color_b, grain));
        }
    }
    pixels
}

fn brick(size: u32, o: &TextureOptions) -> Vec<[u8; 4]> {
    let s = clean_scale(o.scale);
    let bw = (s * 16.0).round().max(4.0) as u32;
    let bh = (s * 8.0).round().max(4.0) as u32;
    let mortar = ((bw.min(bh) as f32 * 0.08).round().max(1.0) as u32)
        .min(bw.saturating_sub(1))
        .min(bh.saturating_sub(1))
        .max(1);

    let mut pixels = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        let row = y / bh;
        let offset = if row % 2 == 0 { 0 } else { bw / 2 };
        for x in 0..size {
            let shifted = x + offset;
            let bx = shifted % bw;
            let by = y % bh;
            if bx < mortar || by < mortar {
                pixels.push(o.color_b);
            } else {
                let col = shifted / bw;
                let variation = 0.82 + 0.18 * lattice(col, row, o.seed);
                pixels.push(shade(o.color_a, variation));
            }
        }
    }
    pixels
}

fn gradient(size: u32, o: &TextureOptions) -> Vec<[u8; 4]> {
    let denom = size.saturating_sub(1).max(1) as f32;
    let mut pixels = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        let t = y as f32 / denom; // 0 at top, 1 at bottom
        let row_color = mix(o.color_a, o.color_b, t);
        for _ in 0..size {
            pixels.push(row_color);
        }
    }
    pixels
}

fn scratches(size: u32, o: &TextureOptions) -> Vec<[u8; 4]> {
    let s = clean_scale(o.scale);
    let cells_y = (s * 32.0).round().max(8.0) as u32;
    let white = [255, 255, 255, o.color_a[3]];
    let black = [0, 0, 0, o.color_a[3]];
    let mut pixels = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        let v = y as f32 / size as f32;
        for x in 0..size {
            let u = x as f32 / size as f32;
            // Long horizontal streaks: few cells across x, many down y.
            let streak = (tileable_fbm(u, v, 3, cells_y, 3, o.seed) - 0.5) * 2.0;
            let detail = tileable_fbm(u, v, 4, cells_y.saturating_mul(2), 2, o.seed ^ 0x5BD1_E3E4);
            let mag = streak.abs();
            if mag > 0.62 {
                let gate = if detail > 0.35 { 1.0 } else { 0.35 };
                let strength = ((mag - 0.62) / 0.38).clamp(0.0, 1.0) * gate;
                let target = if streak > 0.0 { white } else { black };
                pixels.push(mix(o.color_a, target, strength * 0.8));
            } else {
                pixels.push(o.color_a);
            }
        }
    }
    pixels
}

/// Generate a deterministic procedural texture of exactly `size x size` RGBA8
/// pixels. `size` is clamped to [`MAX_TEXTURE_SIZE`] (and at least 1).
pub fn generate(pattern: Pattern, size: u32, options: &TextureOptions) -> TextureData {
    let size = size.clamp(1, MAX_TEXTURE_SIZE);
    let pixels = match pattern {
        Pattern::Checkerboard => checkerboard(size, options),
        Pattern::ValueNoise => value_noise(size, options),
        Pattern::Wood => wood(size, options),
        Pattern::Brick => brick(size, options),
        Pattern::Gradient => gradient(size, options),
        Pattern::Scratches => scratches(size, options),
    };

    TextureData {
        name: format!(
            "procedural_{}_{}x{}_seed{}",
            pattern.as_str(),
            size,
            size,
            options.seed
        ),
        width: size,
        height: size,
        pixels,
    }
}
