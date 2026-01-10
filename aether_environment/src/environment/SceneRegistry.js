export class SceneRegistry {
  constructor(scene) {
    this.scene = scene;
    this.entities = new Map();
  }

  track(name, instance) {
    this.entities.set(name, instance);
  }

  untrack(name) {
    this.entities.delete(name);
  }

  snapshot() {
    const payload = {};
    for (const [name, entity] of this.entities.entries()) {
      if (typeof entity.serialize === "function") {
        payload[name] = entity.serialize();
      }
    }
    return payload;
  }

  find(name) {
    return this.entities.get(name);
  }
}
