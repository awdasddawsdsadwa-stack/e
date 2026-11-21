# Forge

A lightweight, ultra-fast, Blender-inspired 3D modeling, texturing, rigging and animation
desktop application written in Rust, with a built-in Model Context Protocol (MCP) server
so AI clients can inspect and edit the scene in real time.

## Driver constraints

This project deliberately targets **OpenGL 3.3 / OpenGL ES 3.0** compatibility:

- Rendering uses the [`three-d`](https://crates.io/crates/three-d) crate (which wraps `glow`).
- UI uses `egui`.
- No Vulkan, no DirectX 12, no WebGPU, no compute shaders.
- Textures are downscaled to a maximum of 1024x1024 before GPU upload.

## Architecture

| Thread | Responsibility |
| --- | --- |
| Main | `three-d` + `egui` event loop, viewport rendering, camera, gizmos, UI panels, animation playback |
| Background (tokio) | MCP server over stdio (JSON-RPC 2.0), dispatching commands to the main thread over channels |

## Building

Release binaries are built by GitHub Actions on every push to `main` for Windows and Linux,
and uploaded as downloadable artifacts (see the `Actions` tab).

Locally:

```powershell
cargo build --release
```

## Running

```powershell
cargo run --release
```

The GUI opens normally. The MCP server reads JSON-RPC 2.0 messages from **stdin** and writes
responses to **stdout** (one JSON object per line). Therefore stdout is reserved for the
protocol — GUI logs go to stderr.

Example MCP handshake:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"example","version":"1.0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"create_mesh_from_faces","arguments":{"name":"Quad","vertices":[[-1,0,-1],[1,0,-1],[1,0,1],[-1,0,1]],"faces":[[0,1,2,3]]}}}
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"get_scene_hierarchy","arguments":{}}}
```

### Tool groups

- **Measure**: `get_scene_hierarchy`, `get_object_info`, `get_scene_info`, `get_viewport_info`, `get_mesh_topology`, `get_modeling_guide`
- **Build**: `create_mesh_from_faces` (the only geometry creator), `duplicate_node`
- **Blender-style topology**: `extrude_faces`, `inset_faces`, `weld_vertices`, `flip_normals`, `bevel_mesh`
- **Modifiers**: `apply_modifier` (mirror, array, solidify, smooth, triangulate, subdivide, extrude, recalculate normals), `modify_mesh`, `set_dimensions`
- **Materials**: `apply_material`, `set_material_preset`, `generate_procedural_texture`
- **Scene**: `transform_object`, `rename_object`, `delete_object`, `select_object`, `set_object_visibility`, `clear_scene`, `create_light`, `set_light`
- **Camera / rendering**: `set_camera`, `focus_selection`, `render_viewport_snapshot`
- **History**: `undo`, `redo`

Agents should call `get_modeling_guide` first. It documents the Y-up unit-primitive contract,
world-space bounds readback, ground-contact checks, and the Blender-style topology workflow
(`create_mesh_from_faces` → `get_mesh_topology` → `extrude_faces` / `inset_faces`).
Each line of stdout is one compact JSON-RPC response; logs go to stderr.

## Releases

Prebuilt Windows and Linux binaries are published at
https://github.com/awdasddawsdsadwa-stack/e/releases (latest: `v0.2.0`).

## Dynamic GPU adaptation

At startup Forge scans the OpenGL version, renderer string and maximum texture size
through glow. It never requests features beyond OpenGL 3.3, but it clamps every texture
upload to the smaller of 1024px and the driver's reported limit, and reports the detected
profile in the top bar. This keeps the UI on the maintained `egui` backend for legacy
drivers rather than switching to an unmaintained WGPU-based GUI crate.

## Roadmap

- **Phase 1** ✅: OpenGL 3.3 window, orbit/pan/zoom camera, grid floor, egui shell, undo/redo, stdio MCP server.
- **Phase 2** ✅: Custom meshes, auto-UV box projection, image/procedural textures, shading modes,
  world bounds + ground readback, Blender-style tools, multi-angle screenshots. Released as `v0.2.0`.
- **Phase 2.5** (current): Real topology editing (create from faces, extrude/inset faces, weld, bevel)
  so agents model like Blender instead of stacking primitives.
- **Phase 3**: Armatures, bones, auto-skinning, pose mode.
- **Phase 4**: Keyframe timeline, glTF/GLB import/export (viewport snapshots already shipped).
