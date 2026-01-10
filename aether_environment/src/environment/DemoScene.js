import * as THREE from "three";

export class DemoScene {
  constructor(scene, registry) {
    this.scene = scene;
    this.registry = registry;
    this.interactables = [];
  }

  populate() {
    const floor = new THREE.Mesh(
      new THREE.BoxGeometry(40, 0.5, 40),
      new THREE.MeshStandardMaterial({ color: 0x0f111a, metalness: 0.1, roughness: 0.8 })
    );
    floor.position.y = -0.25;
    floor.receiveShadow = true;
    this.scene.add(floor);

    // Simple cubes that Aether can push around later
    for (let i = 0; i < 5; i += 1) {
      const cube = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshStandardMaterial({ color: new THREE.Color().setHSL(Math.random(), 0.5, 0.55) })
      );
      cube.position.set(Math.random() * 10 - 5, 0.5, Math.random() * 10 - 5);
      cube.castShadow = true;
      cube.name = `crate_${i}`;
      this.scene.add(cube);
      this.interactables.push(cube);
    }
  }

  update(dt) {
    // Basic idle animation for set dressing (can be replaced with vehicles or NPCs)
    this.interactables.forEach((mesh, index) => {
      mesh.rotation.y += dt * 0.1 * (index + 1);
    });
  }

  serialize() {
    return this.interactables.map(mesh => ({
      name: mesh.name,
      position: { x: mesh.position.x, y: mesh.position.y, z: mesh.position.z }
    }));
  }
}
