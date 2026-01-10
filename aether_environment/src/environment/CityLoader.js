import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { clone } from "three/examples/jsm/utils/SkeletonUtils.js";

/**
 * Full City Environment Generator
 * Creates a complete city with roads, buildings, parks, and surrounding forest
 * 
 * Grid: 1 unit = 1 tile, scaled up by moduleScale for proper sizing
 */

const TILE = 1;

// Asset files
const A = {
  ROAD: "road-straight.glb",
  ROAD_LIT: "road-straight-lightposts.glb",
  CORNER: "road-corner.glb",
  SPLIT: "road-split.glb",
  CROSS: "road-intersection.glb",
  PAVEMENT: "pavement.glb",
  FOUNTAIN: "pavement-fountain.glb",
  HOUSE_A: "building-small-a.glb",
  HOUSE_B: "building-small-b.glb",
  OFFICE_C: "building-small-c.glb",
  OFFICE_D: "building-small-d.glb",
  GARAGE: "building-garage.glb",
  GRASS: "grass.glb",
  TREES: "grass-trees.glb",
  FOREST: "grass-trees-tall.glb"
};

// Rotation shortcuts
const R0 = 0;
const R90 = Math.PI / 2;
const R180 = Math.PI;
const R270 = (3 * Math.PI) / 2;

/**
 * Generate a small cozy city layout - 4 blocks around a central park
 * Much lighter on performance!
 */
function generateCityLayout() {
  const layout = [];
  let id = 0;
  const add = (file, x, z, rot = 0, scale = 1) => {
    layout.push({ name: `tile_${id++}`, file, position: [x, 0, z], rotation: rot, scale });
  };

  // ═══════════════════════════════════════════════════════════════════════
  // SIMPLE ROAD NETWORK - Just a cross pattern with outer ring
  // ═══════════════════════════════════════════════════════════════════════

  // Main horizontal road (Z = 0) 
  add(A.ROAD_LIT, -3, 0, R90);
  add(A.ROAD_LIT, -2, 0, R90);
  add(A.ROAD_LIT, -1, 0, R90);
  add(A.CROSS, 0, 0, R0);  // Central intersection
  add(A.ROAD_LIT, 1, 0, R90);
  add(A.ROAD_LIT, 2, 0, R90);
  add(A.ROAD_LIT, 3, 0, R90);
  
  // Main vertical road (X = 0)
  add(A.ROAD_LIT, 0, -3, R0);
  add(A.ROAD_LIT, 0, -2, R0);
  add(A.ROAD_LIT, 0, -1, R0);
  // Center already placed
  add(A.ROAD_LIT, 0, 1, R0);
  add(A.ROAD_LIT, 0, 2, R0);
  add(A.ROAD_LIT, 0, 3, R0);

  // Outer ring road (at ±4)
  // Corners
  add(A.CORNER, -4, 4, R180);   // NW
  add(A.CORNER, 4, 4, R270);    // NE
  add(A.CORNER, 4, -4, R0);     // SE
  add(A.CORNER, -4, -4, R90);   // SW

  // North edge
  add(A.ROAD, -3, 4, R90);
  add(A.ROAD, -2, 4, R90);
  add(A.ROAD, -1, 4, R90);
  add(A.SPLIT, 0, 4, R180);  // T-junction connecting to main road
  add(A.ROAD, 1, 4, R90);
  add(A.ROAD, 2, 4, R90);
  add(A.ROAD, 3, 4, R90);
  
  // South edge
  add(A.ROAD, -3, -4, R90);
  add(A.ROAD, -2, -4, R90);
  add(A.ROAD, -1, -4, R90);
  add(A.SPLIT, 0, -4, R0);
  add(A.ROAD, 1, -4, R90);
  add(A.ROAD, 2, -4, R90);
  add(A.ROAD, 3, -4, R90);
  
  // West edge
  add(A.ROAD, -4, -3, R0);
  add(A.ROAD, -4, -2, R0);
  add(A.ROAD, -4, -1, R0);
  add(A.SPLIT, -4, 0, R90);
  add(A.ROAD, -4, 1, R0);
  add(A.ROAD, -4, 2, R0);
  add(A.ROAD, -4, 3, R0);
  
  // East edge
  add(A.ROAD, 4, -3, R0);
  add(A.ROAD, 4, -2, R0);
  add(A.ROAD, 4, -1, R0);
  add(A.SPLIT, 4, 0, R270);
  add(A.ROAD, 4, 1, R0);
  add(A.ROAD, 4, 2, R0);
  add(A.ROAD, 4, 3, R0);

  // ═══════════════════════════════════════════════════════════════════════
  // CENTRAL PARK - Beautiful park with fountain (NW quadrant, where Aether spawns)
  // ═══════════════════════════════════════════════════════════════════════
  
  // Park: X -3 to -1, Z 1 to 3
  add(A.TREES, -3, 3, R0);
  add(A.GRASS, -2, 3, R90);
  add(A.TREES, -1, 3, R180);
  add(A.GRASS, -3, 2, R270);
  add(A.FOUNTAIN, -2, 2, R0);  // ⛲ Beautiful fountain in center!
  add(A.GRASS, -1, 2, R90);
  add(A.TREES, -3, 1, R180);
  add(A.GRASS, -2, 1, R0);
  add(A.TREES, -1, 1, R270);

  // ═══════════════════════════════════════════════════════════════════════
  // BLOCK 1: RESIDENTIAL (SW quadrant) - Cozy homes
  // ═══════════════════════════════════════════════════════════════════════
  
  // X -3 to -1, Z -3 to -1
  add(A.HOUSE_A, -3, -1, R0);
  add(A.HOUSE_B, -2, -1, R90);
  add(A.HOUSE_A, -1, -1, R180);
  add(A.GRASS, -3, -2, R0);
  add(A.TREES, -2, -2, R90);  // Nice tree in the yard
  add(A.GRASS, -1, -2, R180);
  add(A.HOUSE_B, -3, -3, R270);
  add(A.GARAGE, -2, -3, R0);  // 🚗 Garage
  add(A.HOUSE_A, -1, -3, R90);

  // ═══════════════════════════════════════════════════════════════════════
  // BLOCK 2: COMMERCIAL (NE quadrant) - Small shops/offices
  // ═══════════════════════════════════════════════════════════════════════
  
  // X 1 to 3, Z 1 to 3
  add(A.OFFICE_C, 1, 3, R180, 1.2);
  add(A.OFFICE_D, 2, 3, R270, 1.2);
  add(A.OFFICE_C, 3, 3, R0, 1.2);
  add(A.PAVEMENT, 1, 2, R0);
  add(A.PAVEMENT, 2, 2, R90);  // Plaza area
  add(A.PAVEMENT, 3, 2, R180);
  add(A.OFFICE_D, 1, 1, R90, 1.2);
  add(A.OFFICE_C, 2, 1, R180, 1.2);
  add(A.OFFICE_D, 3, 1, R270, 1.2);

  // ═══════════════════════════════════════════════════════════════════════
  // BLOCK 3: MIXED USE (SE quadrant) - Homes + small office
  // ═══════════════════════════════════════════════════════════════════════
  
  // X 1 to 3, Z -3 to -1
  add(A.HOUSE_B, 1, -1, R0);
  add(A.OFFICE_C, 2, -1, R90, 1.1);
  add(A.HOUSE_A, 3, -1, R180);
  add(A.GRASS, 1, -2, R270);
  add(A.FOUNTAIN, 2, -2, R0);  // ⛲ Small neighborhood fountain
  add(A.GRASS, 3, -2, R90);
  add(A.HOUSE_A, 1, -3, R270);
  add(A.HOUSE_B, 2, -3, R0);
  add(A.GARAGE, 3, -3, R90);

  // ═══════════════════════════════════════════════════════════════════════
  // SMALL FOREST BORDER (just a few trees around the edge)
  // ═══════════════════════════════════════════════════════════════════════
  
  // Sparse trees outside the road ring - only in corners
  add(A.FOREST, -5, 5, R0);
  add(A.TREES, -5, -5, R90);
  add(A.FOREST, 5, 5, R180);
  add(A.TREES, 5, -5, R270);
  
  // A few extra trees for ambiance
  add(A.TREES, -6, 0, R0);
  add(A.TREES, 6, 0, R90);
  add(A.TREES, 0, 6, R180);
  add(A.TREES, 0, -6, R270);

  return layout;
}

const DEFAULT_LAYOUT = generateCityLayout();

export class CityLoader {
  constructor(scene, registry, options = {}) {
    this.scene = scene;
    this.registry = registry;
    this.assetsPath = options.assetsPath ?? "/assets/environment/city";
    this.layout = options.layout ?? DEFAULT_LAYOUT;
    this.moduleScale = options.moduleScale ?? 2.5; // Scale up for proper sizing
    this.loader = new GLTFLoader();
    this.cache = new Map();
    this.instances = [];
  }

  async load() {
    await Promise.all(this.layout.map(def => this._placeModule(def)));
  }

  async _placeModule(definition) {
    try {
      const object3d = await this._getAsset(definition.file);
      // Scale positions by moduleScale for proper spacing
      object3d.position.set(
        definition.position[0] * this.moduleScale,
        definition.position[1],
        definition.position[2] * this.moduleScale
      );
      object3d.rotation.y = definition.rotation ?? 0;
      const baseScale = definition.scale ?? 1;
      if (Array.isArray(baseScale)) {
        object3d.scale.set(
          baseScale[0] * this.moduleScale,
          (baseScale[1] ?? baseScale[0]) * this.moduleScale,
          (baseScale[2] ?? baseScale[0]) * this.moduleScale
        );
      } else {
        object3d.scale.setScalar(baseScale * this.moduleScale);
      }

      this.scene.add(object3d);
      this.instances.push(object3d);

      const entity = {
        mesh: object3d,
        serialize: () => ({
          type: definition.file.replace(".glb", ""),
          position: {
            x: object3d.position.x,
            y: object3d.position.y,
            z: object3d.position.z
          },
          rotation: object3d.rotation.y
        }),
        applyProps: props => {
          if (typeof props.enabled === "boolean") {
            object3d.visible = props.enabled;
          }
          if (props.position) {
            object3d.position.set(
              props.position.x ?? object3d.position.x,
              props.position.y ?? object3d.position.y,
              props.position.z ?? object3d.position.z
            );
          }
          if (props.rotation !== undefined) {
            object3d.rotation.y = props.rotation;
          }
        }
      };

      this.registry.track(definition.name, entity);
    } catch (error) {
      console.warn(`Failed to load city module ${definition.file}`, error);
    }
  }

  async _getAsset(file) {
    if (!this.cache.has(file)) {
      const gltf = await this.loader.loadAsync(`${this.assetsPath}/${file}`);
      gltf.scene.traverse(child => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          child.material.side = THREE.DoubleSide;
        }
      });
      this.cache.set(file, gltf.scene);
    }
    return clone(this.cache.get(file));
  }
}
