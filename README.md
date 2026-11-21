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
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"spawn_primitive","arguments":{"primitive":"torus","position":[0,1,0]}}}
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"get_scene_hierarchy","arguments":{}}}
```

Phase 1 tools: `get_scene_hierarchy`, `spawn_primitive`, `transform_object`, `undo`, `redo`.
Each line of stdout is one compact JSON-RPC response; logs go to stderr.

## Roadmap

- **Phase 1** (current): OpenGL 3.3 window, orbit/pan/zoom camera, grid floor, egui shell
  (Outliner / Inspector / Timeline), undo/redo foundation, stdio MCP server with
  `spawn_primitive`, `transform_object` and `get_scene_hierarchy`.
- **Phase 2**: Custom meshes, auto-UV box projection, image/procedural textures, shading modes.
- **Phase 3**: Armatures, bones, auto-skinning, pose mode.
- **Phase 4**: Keyframe timeline, glTF/GLB import/export, viewport snapshot feedback.
