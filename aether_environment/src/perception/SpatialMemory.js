/**
 * SpatialMemory.js
 * 
 * Aether's spatial memory system - remembers locations, landmarks, and exploration history.
 * Enables intelligent exploration by tracking:
 * - Visited locations (grid-based)
 * - Discovered landmarks (fountains, buildings, etc.)
 * - Paths taken
 * - Points of interest
 */

import * as THREE from "three";

// Grid cell size for spatial memory (world units)
const CELL_SIZE = 2.0;

// Landmark types with associated keywords for detection
const LANDMARK_TYPES = {
  fountain: { keywords: ["fountain"], icon: "⛲", interest: 10 },
  park: { keywords: ["grass", "trees", "forest"], icon: "🌳", interest: 6 },
  building: { keywords: ["building", "house", "office", "garage"], icon: "🏠", interest: 5 },
  road: { keywords: ["road", "pavement", "intersection", "corner"], icon: "🛤️", interest: 2 },
  plaza: { keywords: ["plaza", "pavement"], icon: "🏛️", interest: 7 }
};

export class SpatialMemory {
  constructor(options = {}) {
    this.options = {
      cellSize: options.cellSize ?? CELL_SIZE,
      maxMemories: options.maxMemories ?? 500,
      explorationRadius: options.explorationRadius ?? 3.0, // How close to mark as "visited"
      ...options
    };
    
    // Grid-based visited tracking (key: "x,z" -> visit data)
    this.visitedCells = new Map();
    
    // Discovered landmarks
    this.landmarks = new Map(); // id -> landmark data
    
    // Path history (ordered list of positions)
    this.pathHistory = [];
    this.maxPathHistory = 200;
    
    // Current exploration state
    this.explorationStats = {
      totalCellsVisited: 0,
      totalLandmarksFound: 0,
      totalDistanceTraveled: 0,
      explorationStartTime: Date.now(),
      lastPosition: null
    };
    
    // World bounds (auto-detected from landmarks)
    this.worldBounds = {
      minX: -15, maxX: 15,
      minZ: -15, maxZ: 15
    };
    
    console.log("🧠 Spatial memory initialized");
  }
  
  /**
   * Get grid cell key from world position
   */
  getCellKey(x, z) {
    const cellX = Math.floor(x / this.options.cellSize);
    const cellZ = Math.floor(z / this.options.cellSize);
    return `${cellX},${cellZ}`;
  }
  
  /**
   * Get world position from cell key
   */
  getCellCenter(cellKey) {
    const [cx, cz] = cellKey.split(",").map(Number);
    return {
      x: (cx + 0.5) * this.options.cellSize,
      z: (cz + 0.5) * this.options.cellSize
    };
  }
  
  /**
   * Mark current position as visited
   */
  visit(position) {
    const { x, y, z } = position;
    const cellKey = this.getCellKey(x, z);
    
    // Update distance traveled
    if (this.explorationStats.lastPosition) {
      const last = this.explorationStats.lastPosition;
      const dist = Math.sqrt((x - last.x) ** 2 + (z - last.z) ** 2);
      this.explorationStats.totalDistanceTraveled += dist;
    }
    this.explorationStats.lastPosition = { x, y, z };
    
    // Mark cell as visited
    if (!this.visitedCells.has(cellKey)) {
      this.visitedCells.set(cellKey, {
        firstVisit: Date.now(),
        lastVisit: Date.now(),
        visitCount: 1,
        center: this.getCellCenter(cellKey)
      });
      this.explorationStats.totalCellsVisited++;
    } else {
      const cell = this.visitedCells.get(cellKey);
      cell.lastVisit = Date.now();
      cell.visitCount++;
    }
    
    // Add to path history
    this.pathHistory.push({
      x, y, z,
      time: Date.now(),
      cell: cellKey
    });
    
    // Trim path history
    if (this.pathHistory.length > this.maxPathHistory) {
      this.pathHistory.shift();
    }
  }
  
  /**
   * Check if a position has been visited
   */
  hasVisited(x, z) {
    const cellKey = this.getCellKey(x, z);
    return this.visitedCells.has(cellKey);
  }
  
  /**
   * Get visit count for a cell
   */
  getVisitCount(x, z) {
    const cellKey = this.getCellKey(x, z);
    const cell = this.visitedCells.get(cellKey);
    return cell ? cell.visitCount : 0;
  }
  
  /**
   * Register a discovered landmark
   */
  discoverLandmark(id, type, position, name = null) {
    if (this.landmarks.has(id)) {
      // Update existing landmark
      const existing = this.landmarks.get(id);
      existing.lastSeen = Date.now();
      existing.seenCount++;
      return existing;
    }
    
    // Get landmark type info
    const typeInfo = LANDMARK_TYPES[type] || { icon: "📍", interest: 3 };
    
    const landmark = {
      id,
      type,
      name: name || `${typeInfo.icon} ${type}`,
      position: { ...position },
      discoveredAt: Date.now(),
      lastSeen: Date.now(),
      seenCount: 1,
      interest: typeInfo.interest,
      icon: typeInfo.icon,
      visited: false
    };
    
    this.landmarks.set(id, landmark);
    this.explorationStats.totalLandmarksFound++;
    
    console.log(`🗺️ Discovered: ${landmark.name} at (${position.x.toFixed(1)}, ${position.z.toFixed(1)})`);
    
    return landmark;
  }
  
  /**
   * Mark a landmark as visited
   */
  visitLandmark(id) {
    const landmark = this.landmarks.get(id);
    if (landmark) {
      landmark.visited = true;
      landmark.lastSeen = Date.now();
    }
  }
  
  /**
   * Get nearest unvisited landmark
   */
  getNearestUnvisitedLandmark(currentPos) {
    let nearest = null;
    let nearestDist = Infinity;
    
    for (const [id, landmark] of this.landmarks) {
      if (landmark.visited) continue;
      
      const dist = Math.sqrt(
        (landmark.position.x - currentPos.x) ** 2 +
        (landmark.position.z - currentPos.z) ** 2
      );
      
      if (dist < nearestDist) {
        nearestDist = dist;
        nearest = landmark;
      }
    }
    
    return nearest ? { landmark: nearest, distance: nearestDist } : null;
  }
  
  /**
   * Get most interesting unexplored direction
   */
  getBestExplorationDirection(currentPos) {
    const directions = [
      { name: "north", dx: 0, dz: 1 },
      { name: "south", dx: 0, dz: -1 },
      { name: "east", dx: 1, dz: 0 },
      { name: "west", dx: -1, dz: 0 },
      { name: "northeast", dx: 0.7, dz: 0.7 },
      { name: "northwest", dx: -0.7, dz: 0.7 },
      { name: "southeast", dx: 0.7, dz: -0.7 },
      { name: "southwest", dx: -0.7, dz: -0.7 }
    ];
    
    let bestDirection = null;
    let bestScore = -Infinity;
    
    for (const dir of directions) {
      let score = 0;
      
      // Check cells in this direction (3 cells out)
      for (let dist = 1; dist <= 3; dist++) {
        const checkX = currentPos.x + dir.dx * dist * this.options.cellSize;
        const checkZ = currentPos.z + dir.dz * dist * this.options.cellSize;
        
        // Within bounds?
        if (checkX < this.worldBounds.minX || checkX > this.worldBounds.maxX ||
            checkZ < this.worldBounds.minZ || checkZ > this.worldBounds.maxZ) {
          score -= 10;
          continue;
        }
        
        // Unvisited cells are more interesting
        const visits = this.getVisitCount(checkX, checkZ);
        if (visits === 0) {
          score += 10 / dist; // Closer unexplored = better
        } else {
          score -= visits * 2; // Avoid revisiting
        }
      }
      
      if (score > bestScore) {
        bestScore = score;
        bestDirection = dir;
      }
    }
    
    return bestDirection;
  }
  
  /**
   * Get unexplored areas as target positions
   */
  getUnexploredTargets(currentPos, count = 5) {
    const targets = [];
    
    // Scan the world grid for unexplored cells
    for (let x = this.worldBounds.minX; x <= this.worldBounds.maxX; x += this.options.cellSize) {
      for (let z = this.worldBounds.minZ; z <= this.worldBounds.maxZ; z += this.options.cellSize) {
        if (!this.hasVisited(x, z)) {
          const dist = Math.sqrt((x - currentPos.x) ** 2 + (z - currentPos.z) ** 2);
          targets.push({ x, z, distance: dist });
        }
      }
    }
    
    // Sort by distance and return closest unexplored
    targets.sort((a, b) => a.distance - b.distance);
    return targets.slice(0, count);
  }
  
  /**
   * Calculate exploration progress
   * Returns object with detailed progress stats
   */
  getExplorationProgress() {
    const totalCells = Math.ceil((this.worldBounds.maxX - this.worldBounds.minX) / this.options.cellSize) *
                       Math.ceil((this.worldBounds.maxZ - this.worldBounds.minZ) / this.options.cellSize);
    const percentage = Math.min(1.0, this.explorationStats.totalCellsVisited / totalCells);
    const visitedLandmarks = [...this.landmarks.values()].filter(l => l.visited).length;
    
    return {
      percentage,
      percentageText: (percentage * 100).toFixed(1) + "%",
      visitedCells: this.explorationStats.totalCellsVisited,
      totalCells,
      discoveredLandmarks: this.landmarks.size,
      visitedLandmarks,
      distanceTraveled: this.explorationStats.totalDistanceTraveled,
      timeExploring: Date.now() - this.explorationStats.explorationStartTime
    };
  }
  
  /**
   * Get a summary for the AI
   */
  getSummary() {
    const progress = this.getExplorationProgress();
    
    return {
      ...progress,
      explorationProgress: progress.percentageText,
      landmarksFound: progress.discoveredLandmarks,
      distanceTraveledText: progress.distanceTraveled.toFixed(1) + " units",
      timeExploringText: Math.floor(progress.timeExploring / 1000) + "s"
    };
  }
  
  /**
   * Describe current exploration state in natural language
   */
  describe(currentPos) {
    const parts = [];
    const progress = this.getExplorationProgress();
    
    parts.push(`I have explored ${(progress.percentage * 100).toFixed(0)}% of the city`);
    
    // Nearby landmarks
    const nearbyLandmarks = [];
    for (const [id, landmark] of this.landmarks) {
      const dist = Math.sqrt(
        (landmark.position.x - currentPos.x) ** 2 +
        (landmark.position.z - currentPos.z) ** 2
      );
      if (dist < 8) {
        nearbyLandmarks.push({ ...landmark, distance: dist });
      }
    }
    
    if (nearbyLandmarks.length > 0) {
      nearbyLandmarks.sort((a, b) => a.distance - b.distance);
      const nearest = nearbyLandmarks[0];
      const direction = this._getDirection(currentPos, nearest.position);
      parts.push(`The ${nearest.name} is ${nearest.distance.toFixed(1)} units to the ${direction}`);
    }
    
    // Suggest exploration
    const bestDir = this.getBestExplorationDirection(currentPos);
    if (bestDir) {
      parts.push(`Unexplored areas lie to the ${bestDir.name}`);
    }
    
    return parts.join(". ") + ".";
  }
  
  /**
   * Get compass direction from one point to another
   */
  _getDirection(from, to) {
    const dx = to.x - from.x;
    const dz = to.z - from.z;
    const angle = Math.atan2(dx, dz) * (180 / Math.PI);
    
    if (angle > -22.5 && angle <= 22.5) return "north";
    if (angle > 22.5 && angle <= 67.5) return "northeast";
    if (angle > 67.5 && angle <= 112.5) return "east";
    if (angle > 112.5 && angle <= 157.5) return "southeast";
    if (angle > 157.5 || angle <= -157.5) return "south";
    if (angle > -157.5 && angle <= -112.5) return "southwest";
    if (angle > -112.5 && angle <= -67.5) return "west";
    if (angle > -67.5 && angle <= -22.5) return "northwest";
    return "nearby";
  }
  
  /**
   * Serialize for backend/storage
   */
  serialize() {
    return {
      visitedCells: Array.from(this.visitedCells.entries()),
      landmarks: Array.from(this.landmarks.entries()),
      explorationStats: this.explorationStats,
      worldBounds: this.worldBounds
    };
  }
  
  /**
   * Load from serialized data
   */
  deserialize(data) {
    if (data.visitedCells) {
      this.visitedCells = new Map(data.visitedCells);
    }
    if (data.landmarks) {
      this.landmarks = new Map(data.landmarks);
    }
    if (data.explorationStats) {
      this.explorationStats = { ...this.explorationStats, ...data.explorationStats };
    }
    if (data.worldBounds) {
      this.worldBounds = data.worldBounds;
    }
  }
}
