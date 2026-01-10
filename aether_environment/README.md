# Aether Environment Sandbox

Purpose-built WebGL playground for training and testing Aether's embodied skills without relying on third-party editors. Drop in Mixamo rigs, Quaternius animation clips, and custom GLTF scenes to give the agent a realistic sensory/motor loop before we transition to humanoid hardware.

## Features

- **Three.js scene graph** with lighting, nav grid, and asset registry
- **Aether bridge** mirrors the PlayCanvas workflow (state sync + command execution)
- **Player rig loader** ready for Mixamo humanoids plus reusable animation states (idle, walk, run)
- **Object interaction hooks** so Aether can move props, toggle doors, or spawn vehicles
- **Vite dev server** for instant reloads while you iterate on assets or scripts

## Getting Started

```bash
cd aether_environment
npm install
npm run dev
```

The dev server defaults to http://localhost:5173. Update `.env.local` if your Aether backend is hosted elsewhere.

## Folder Layout

```
aether_environment/
├── public/
│   └── assets/            # Place GLTF/GLB, textures, audio, etc.
├── src/
│   ├── main.js            # bootstrap + render loop
│   ├── services/
│   │   └── AetherBridge.js
│   ├── player/
│   │   └── PlayerRig.js
│   ├── environment/
│   │   ├── DemoScene.js
│   │   └── SceneRegistry.js
│   └── utils/
│       └── controls.js
├── package.json
├── vite.config.js
├── README.md
└── .env.example
```

## Importing Assets

1. Download the **Quaternius Universal Animation Library** (`UniversalAnimations.glb`) and drop it under `public/assets/animations/`.
2. Export your Mixamo character as `T-pose` FBX/GLB → convert to GLB (if needed) using Blender → place it under `public/assets/characters/`.
3. Update the paths in `PlayerRig.js` (see `CHARACTER_ASSET` and `ANIMATION_LIBRARY`) to match your filenames.
4. Add additional props/vehicles to `public/assets/props/` and register them through `DemoScene` or custom loaders.

## Wiring to Aether

- `.env.example` contains `VITE_AETHER_API_URL` and `VITE_AETHER_API_KEY` placeholders. Copy it to `.env.local` and set real values.
- The bridge posts scene telemetry every 500 ms (configurable) and executes returned commands (`move`, `spawn`, `set_property`, etc.) exactly like the PlayCanvas integration.
- Extend `SceneRegistry` if you need extra sensor channels (e.g., LiDAR depth, synthetic segmentation masks, surface friction curves).

## Next Steps

- Add rapier.js or cannon-es for physically-accurate interactions
- Stream rendered camera frames back to the backend for computer-vision training
- Expand the animation state machine (jump, crouch, push, pull)
- Procedurally generate training mazes so Aether can learn navigation priors

Happy building. When you are ready for hardware, swap the bridge target to the humanoid adapter—the control contract stays the same.
