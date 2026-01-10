/**
 * PhysicsWorld.js
 * Rapier.js physics simulation for Aether's embodied experience
 * Provides realistic collision, ragdoll, and environmental interaction
 */

import RAPIER from "@dimforge/rapier3d-compat";

let rapierReady = false;
let RAPIER_INSTANCE = null;

// Initialize Rapier (must be called before creating PhysicsWorld)
export async function initRapier() {
  if (rapierReady) return RAPIER_INSTANCE;
  await RAPIER.init();
  RAPIER_INSTANCE = RAPIER;
  rapierReady = true;
  console.log("⚡ Rapier physics engine initialized");
  return RAPIER;
}

export class PhysicsWorld {
  constructor(scene) {
    if (!rapierReady) {
      throw new Error("Call initRapier() before creating PhysicsWorld");
    }
    
    this.scene = scene;
    this.RAPIER = RAPIER_INSTANCE;
    
    // Create physics world with gravity
    this.world = new this.RAPIER.World({ x: 0, y: -9.81, z: 0 });
    
    // Track physics bodies and their Three.js counterparts
    this.bodies = new Map(); // rigidBody -> { mesh, type }
    this.colliders = new Map();
    
    // Debug visualization
    this.debugLines = null;
    this.debugEnabled = false;
    
    console.log("🌍 Physics world created");
  }
  
  /**
   * Create a static ground plane
   */
  createGround(size = 100, y = 0) {
    const groundDesc = this.RAPIER.RigidBodyDesc.fixed()
      .setTranslation(0, y - 0.25, 0);
    const groundBody = this.world.createRigidBody(groundDesc);
    
    const groundColliderDesc = this.RAPIER.ColliderDesc.cuboid(size / 2, 0.25, size / 2);
    this.world.createCollider(groundColliderDesc, groundBody);
    
    return groundBody;
  }
  
  /**
   * Create a dynamic physics body for an object
   */
  createDynamicBody(mesh, options = {}) {
    const {
      mass = 1,
      friction = 0.5,
      restitution = 0.3,
      shape = "auto",
      isCCD = false // Continuous collision detection for fast objects
    } = options;
    
    const pos = mesh.position;
    const rot = mesh.quaternion;
    
    const bodyDesc = this.RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(pos.x, pos.y, pos.z)
      .setRotation({ x: rot.x, y: rot.y, z: rot.z, w: rot.w })
      .setCcdEnabled(isCCD);
    
    const body = this.world.createRigidBody(bodyDesc);
    
    // Auto-detect collider shape from geometry
    let colliderDesc;
    if (shape === "auto" && mesh.geometry) {
      mesh.geometry.computeBoundingBox();
      const box = mesh.geometry.boundingBox;
      const size = {
        x: (box.max.x - box.min.x) * mesh.scale.x / 2,
        y: (box.max.y - box.min.y) * mesh.scale.y / 2,
        z: (box.max.z - box.min.z) * mesh.scale.z / 2
      };
      colliderDesc = this.RAPIER.ColliderDesc.cuboid(size.x, size.y, size.z);
    } else if (shape === "sphere") {
      const radius = mesh.geometry?.parameters?.radius || 0.5;
      colliderDesc = this.RAPIER.ColliderDesc.ball(radius * mesh.scale.x);
    } else if (shape === "capsule") {
      const height = options.height || 1.0;
      const radius = options.radius || 0.3;
      colliderDesc = this.RAPIER.ColliderDesc.capsule(height / 2, radius);
    } else {
      // Default to unit cube
      colliderDesc = this.RAPIER.ColliderDesc.cuboid(0.5, 0.5, 0.5);
    }
    
    colliderDesc.setMass(mass)
      .setFriction(friction)
      .setRestitution(restitution);
    
    const collider = this.world.createCollider(colliderDesc, body);
    
    this.bodies.set(body, { mesh, type: "dynamic" });
    this.colliders.set(collider, mesh);
    
    return { body, collider };
  }
  
  /**
   * Create a kinematic body (AI-controlled, affects physics but not affected)
   */
  createKinematicBody(mesh, options = {}) {
    const pos = mesh.position;
    const rot = mesh.quaternion;
    
    const bodyDesc = this.RAPIER.RigidBodyDesc.kinematicPositionBased()
      .setTranslation(pos.x, pos.y, pos.z)
      .setRotation({ x: rot.x, y: rot.y, z: rot.z, w: rot.w });
    
    const body = this.world.createRigidBody(bodyDesc);
    
    // Capsule collider for character
    const height = options.height || 0.7;
    const radius = options.radius || 0.15;
    const colliderDesc = this.RAPIER.ColliderDesc.capsule(height / 2, radius)
      .setTranslation(0, height / 2 + radius, 0);
    
    const collider = this.world.createCollider(colliderDesc, body);
    
    this.bodies.set(body, { mesh, type: "kinematic" });
    
    return { body, collider };
  }
  
  /**
   * Update kinematic body position (for AI-controlled movement)
   */
  setKinematicPosition(body, position, rotation) {
    body.setNextKinematicTranslation({ x: position.x, y: position.y, z: position.z });
    if (rotation !== undefined) {
      body.setNextKinematicRotation({ x: 0, y: Math.sin(rotation / 2), z: 0, w: Math.cos(rotation / 2) });
    }
  }
  
  /**
   * Apply force to a dynamic body
   */
  applyForce(body, force, point = null) {
    if (point) {
      body.applyImpulseAtPoint(force, point, true);
    } else {
      body.applyImpulse(force, true);
    }
  }
  
  /**
   * Cast a ray and return hit information
   */
  raycast(origin, direction, maxDistance = 100) {
    const ray = new this.RAPIER.Ray(origin, direction);
    const hit = this.world.castRay(ray, maxDistance, true);
    
    if (hit) {
      const hitPoint = ray.pointAt(hit.toi);
      const collider = hit.collider;
      const mesh = this.colliders.get(collider);
      
      return {
        distance: hit.toi,
        point: hitPoint,
        normal: hit.normal,
        mesh: mesh,
        collider: collider
      };
    }
    return null;
  }
  
  /**
   * Check what's around a point (for spatial awareness)
   */
  queryNearby(position, radius = 2) {
    const shape = new this.RAPIER.Ball(radius);
    const shapePos = { x: position.x, y: position.y, z: position.z };
    const shapeRot = { x: 0, y: 0, z: 0, w: 1 };
    
    const nearby = [];
    this.world.intersectionsWithShape(shapePos, shapeRot, shape, (collider) => {
      const mesh = this.colliders.get(collider);
      if (mesh) {
        nearby.push({
          mesh: mesh,
          name: mesh.name || "unknown",
          distance: Math.sqrt(
            Math.pow(mesh.position.x - position.x, 2) +
            Math.pow(mesh.position.y - position.y, 2) +
            Math.pow(mesh.position.z - position.z, 2)
          )
        });
      }
      return true; // Continue searching
    });
    
    return nearby.sort((a, b) => a.distance - b.distance);
  }
  
  /**
   * Step the physics simulation
   */
  step(dt) {
    // Use fixed timestep for stability
    this.world.timestep = Math.min(dt, 1 / 30);
    this.world.step();
    
    // Sync Three.js meshes with physics bodies
    for (const [body, data] of this.bodies) {
      if (data.type === "dynamic") {
        const pos = body.translation();
        const rot = body.rotation();
        
        data.mesh.position.set(pos.x, pos.y, pos.z);
        data.mesh.quaternion.set(rot.x, rot.y, rot.z, rot.w);
      }
    }
  }
  
  /**
   * Get collision events from this frame
   */
  getCollisionEvents() {
    const events = [];
    this.world.contactPairsWith(null, (collider1, collider2, started) => {
      events.push({
        collider1: this.colliders.get(collider1),
        collider2: this.colliders.get(collider2),
        started: started
      });
    });
    return events;
  }
  
  /**
   * Clean up
   */
  destroy() {
    this.world.free();
    this.bodies.clear();
    this.colliders.clear();
  }
}
