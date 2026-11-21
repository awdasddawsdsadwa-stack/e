//! Runtime GPU capability scan.
//!
//! We keep a single renderer/UI backend (three-d + egui, OpenGL 3.3 Core), but
//! detect the actual driver at startup and adapt the safe resource limits.
//! This is the dynamic fallback path for older GPUs instead of adding a
//! second, WGPU-based UI backend (which would not run on legacy drivers).

use std::ops::Deref;
use three_d::context as gl;
use three_d::context::HasContext as _;

#[derive(Clone, Debug)]
pub struct GpuProfile {
    pub version_string: String,
    pub renderer: String,
    pub major: u32,
    pub minor: u32,
    pub driver_max_texture: u32,
    pub low_end: bool,
    pub texture_limit: u32,
}

fn parse_version(text: &str) -> (u32, u32) {
    for token in text.split_whitespace() {
        if !token.as_bytes().first().is_some_and(u8::is_ascii_digit) {
            continue;
        }
        let parts: Vec<&str> = token.split('.').collect();
        if let (Some(major), Some(minor)) = (parts.first(), parts.get(1)) {
            if let (Ok(major), Ok(minor)) = (major.parse::<u32>(), minor.parse::<u32>()) {
                return (major, minor);
            }
        }
    }
    (3, 3)
}

impl GpuProfile {
    pub fn detect(context: &three_d::Context) -> Self {
        // three-d's mid-level Context dereferences to Arc<glow::Context>.
        let glow_ctx: &gl::Context = context.deref().as_ref();
        let version_string = unsafe { glow_ctx.get_parameter_string(gl::VERSION) };
        let renderer = unsafe { glow_ctx.get_parameter_string(gl::RENDERER) };
        let driver_max_texture = unsafe { glow_ctx.get_parameter_i32(gl::MAX_TEXTURE_SIZE) }.max(1) as u32;
        let (major, minor) = parse_version(&version_string);

        // GL 3.3 (and GLES 3.0/3.1) are the legacy path. We never enable a
        // feature that requires 4.3+, regardless of what the driver reports.
        let low_end = major < 4 || (major == 4 && minor < 3);
        let texture_limit = driver_max_texture.min(crate::texture::MAX_TEXTURE_SIZE);

        Self {
            version_string,
            renderer,
            major,
            minor,
            driver_max_texture,
            low_end,
            texture_limit,
        }
    }

    pub fn summary(&self) -> String {
        let tier = if self.low_end { "legacy" } else { "modern" };
        format!("{} GL {}.{} ({}), textures <= {}px", tier, self.major, self.minor, self.renderer, self.texture_limit)
    }
}
