# Portfolio Project Data — Autonomous Warehouse Navigation with Deep Q-Learning

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/autonomous_warehouse_navigation_with/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image08.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "autonomous-warehouse-navigation-with-deep-q-learning",
  "code": "PRO-32 // ML-17",
  "title": "Autonomous Warehouse Navigation with Deep Q-Learning",
  "category": "machine-learning",
  "categoryLabel": "Reinforcement Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Autonomous-Warehouse-Navigation-with-Deep-Q-Learning",
  "license": "MIT License",
  "banner": "/images/projects/autonomous_warehouse_navigation_with/banner.webp",
  "summary": "PyTorch Deep Q-Network AGV navigation system achieving optimal 22-step paths and 0.15ms inference against BFS and A* warehouse baselines.",
  "projectDescription": "A PyTorch Deep Q-Learning system for autonomous warehouse AGV navigation across 2D grid environments. Combining a 19-dimensional spatial sensor state, experience replay, and Polyak target networks, it achieves optimal 22-step collision-free routing in 0.15ms inference.",
  "architectureSummary": "2D Warehouse Grid -> 19-Dim Normalized State Vector -> MLP Policy & Polyak Target DQN -> Greedy Evaluation vs BFS & A* -> Streamlit Simulator",
  "techStack": [
    "Python",
    "PyTorch",
    "NumPy",
    "Pandas",
    "Matplotlib",
    "Pillow",
    "ImageIO",
    "Streamlit"
  ],
  "telemetryMatrix": {
    "projectType": "Deep Reinforcement Learning Navigation",
    "targetPlatform": "Python PyTorch & Streamlit Dashboard",
    "primaryDomain": "Autonomous Warehouse Robotics",
    "architecturePipeline": "19-Dim Obs -> MLP DQN -> Greedy Eval",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Reinforcement Learning Pathfinding Agent",
    "learningParadigm": "Reinforcement Learning",
    "modelDomain": "Autonomous Warehouse Robotics",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Deep Q-Learning",
      "19-Dim State Perception",
      "BFS A* Benchmarking"
    ]
  },
  "mlProblemContext": "Autonomous Guided Vehicles in fulfillment centers require fast, collision-free routing around storage racks. Unlike classical BFS and A* planners that demand complete global map matrices, this project trains a Deep Q-Network to learn optimal navigation from compact local and relative observations.",
  "mlDatasetOverview": {
    "profileBadge": "Synthetic 2D Grid & CSV Logs",
    "source": "environment/maps.py, dataset/training_history.csv, dataset/evaluation_results.csv",
    "modality": "2D Spatial Grid & Numerical State Vectors",
    "scale": "600 Training Episodes + 4 Maps x 30 Eval Rollouts",
    "inputShape": "19 Normalized Float32 Features",
    "dataFormat": "NumPy Arrays & CSV Logs",
    "description": "Dynamic interaction transitions generated across four verified 2D grid maps (10x10 simple, 12x12 moderate, 12x12 complex, and 12x12 generalization), with per-episode training metrics and multi-map evaluation benchmarks logged to CSV.",
    "provenanceNote": "Self-contained synthetic warehouse layouts validated for path solvability via BFS before simulation."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "19-Dim State & 4-Action Space",
    "spaceNote": "Size-invariant continuous observation vector paired with 4 discrete cardinal moves",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Discrete Action Q-Values (UP, DOWN, LEFT, RIGHT)",
      "type": "4-Action Discrete Value Estimates",
      "distribution": "Optimal greedy policy selected via argmax Q(s, a)"
    },
    "inputSignals": [
      {
        "name": "Normalized Robot & Goal Coordinates",
        "type": "Numerical Continuous (4 dims)",
        "importanceOrRole": "Global Positional Awareness",
        "description": "Row and column positions of the AGV and drop-off bay scaled by grid height and width minus one."
      },
      {
        "name": "Relative Displacement & Manhattan Distance",
        "type": "Numerical Continuous (3 dims)",
        "importanceOrRole": "Target Vector Guidance",
        "description": "Signed row/column delta toward the goal plus normalized L1 Manhattan distance."
      },
      {
        "name": "Local 3x3 Occupancy Mask",
        "type": "Binary Spatial Sensor (8 dims)",
        "importanceOrRole": "Immediate Collision Avoidance",
        "description": "Eight-neighbor binary flags encoding adjacent storage racks and boundary walls around the agent."
      },
      {
        "name": "Cardinal Ray-Casting Clearance",
        "type": "Numerical Sensor (4 dims)",
        "importanceOrRole": "Aisle Depth Perception",
        "description": "Normalized free-cell line-of-sight distances along UP, DOWN, LEFT, and RIGHT directions."
      }
    ],
    "dataCharacteristicsSummary": "All 19 input features are normalized to continuous [0.0, 1.0] or [-1.0, 1.0] scales, decoupling the MLP policy from fixed grid sizes. Across 600 episodes on moderate_map, episode rewards converge from -261.0 to +66.6 as collisions drop to zero and path length shrinks to the 22-step optimum."
  },
  "mlLearningApproach": {
    "paradigm": "Off-Policy Value-Based Reinforcement Learning (Markov Decision Process)",
    "methodologyDescription": "The navigation problem is formulated as a discrete Markov Decision Process solved via off-policy Deep Q-Learning. A three-hidden-layer PyTorch MLP maps 19-dimensional spatial vectors to four directional Q-values, guided by potential-based Manhattan distance shaping and Huber Bellman updates.",
    "justification": "DQN delivers constant-time O(1) forward-pass inference without requiring full global obstacle matrices at runtime, while potential-based distance shaping accelerates convergence without altering optimal shortest-path policies."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Grid Layout Solvability Verification",
      "description": "Validate every warehouse rack matrix in environment/maps.py using bfs_search to ensure at least one collision-free path exists from start to goal.",
      "badge": "Map Check"
    },
    {
      "step": "02",
      "title": "Spatial Feature Normalization",
      "description": "Scale robot coordinates, goal coordinates, relative displacements, and Manhattan distance to grid-invariant ranges in [0.0, 1.0] and [-1.0, 1.0].",
      "badge": "Normalization"
    },
    {
      "step": "03",
      "title": "Local Sensor & Ray-Cast Encoding",
      "description": "Extract 8-directional 3x3 obstacle occupancy flags and 4-way cardinal ray-casting clearance distances into a 19-dim float32 tensor.",
      "badge": "Feature Vector"
    },
    {
      "step": "04",
      "title": "Circular Replay Buffer Decorrelation",
      "description": "Store (s, a, r, s', done) tuples in a 40,000-capacity deque and draw uniform 64-transition mini-batches to break temporal correlation.",
      "badge": "Experience Replay"
    }
  ],
  "mlDataPreparationNote": "Training occurs exclusively on moderate_map while simple_map, complex_map, and generalization_map are strictly held out for zero-shot deterministic evaluation.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Deep Q-Network (MLP QNetwork + Polyak Target Network)",
    "architectureSummary": "Fully connected Multi-Layer Perceptron (19 -> 128 -> 128 -> 64 -> 4) with ReLU hidden activations and linear Q-value outputs, paired with an identical target network updated via soft Polyak averaging.",
    "configurationBadge": "PyTorch DQN Config",
    "hyperparameters": [
      {
        "name": "hidden_dims",
        "value": "(128, 128, 64)",
        "description": "Three fully connected hidden layers with ReLU activations."
      },
      {
        "name": "learning_rate",
        "value": "0.0005",
        "description": "Adam step size with gradient norm clipping at 5.0."
      },
      {
        "name": "gamma",
        "value": "0.99",
        "description": "Discount factor prioritizing long-horizon goal completion."
      },
      {
        "name": "replay_capacity / batch_size",
        "value": "40000 / 64",
        "description": "Circular experience buffer size and uniform mini-batch size."
      },
      {
        "name": "epsilon_schedule",
        "value": "1.0 -> 0.05 (0.994)",
        "description": "Per-episode exponential decay for epsilon-greedy exploration."
      },
      {
        "name": "target_update_tau",
        "value": "0.005 (every 10 steps)",
        "description": "Soft Polyak blending coefficient for target network stabilization."
      }
    ],
    "objectiveCriterion": "Smooth L1 Loss (Huber Loss) on TD Bellman Targets",
    "optimizationSolver": "Adam Optimizer with Gradient Norm Clipping (max norm 5.0)",
    "trainingProcess": "Training runs for 600 episodes on moderate_map using seed 42, completing in 42.18s on CPU. Epsilon decays exponentially from 1.0 to 0.05 while mini-batches of 64 transitions update the policy network via Adam, reaching 100% 50-episode moving success at episode 126 and +64.9 peak moving reward."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Deterministic Greedy Rollouts vs BFS & A* Across 4 Warehouse Maps",
    "description": "After training on moderate_map, models/best_model.pth is evaluated with exploration disabled (epsilon = 0.0) over 30 independent rollouts per scenario on simple_map, moderate_map, complex_map, and generalization_map against classical BFS and A* baselines.",
    "metrics": [
      {
        "label": "MODERATE MAP SUCCESS",
        "value": "100.0%",
        "benchmark": "Target >= 95.0%"
      },
      {
        "label": "MODERATE PATH LENGTH",
        "value": "22.0 steps",
        "benchmark": "Optimal = 22.0"
      },
      {
        "label": "SIMPLE MAP ZERO-SHOT",
        "value": "18.0 steps",
        "benchmark": "Optimal = 18.0"
      },
      {
        "label": "COLLISIONS (MOD/SIMPLE)",
        "value": "0.00 / ep",
        "benchmark": "Target < 0.50"
      },
      {
        "label": "INFERENCE LATENCY",
        "value": "0.150 ms",
        "benchmark": "Baseline = 0.152ms"
      },
      {
        "label": "PEAK 50-EP REWARD",
        "value": "+64.90",
        "benchmark": "Target > 60.0"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "System Overview Dashboard",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/autonomous_warehouse_navigation_with/image01.webp",
      "telemetryNote": "Displays 100.0% success rate, 0.00 collision rate, 100% path efficiency, 0.15ms latency, and the 12x12 moderate_map layout."
    },
    {
      "title": "Interactive AGV Navigation Simulator",
      "badge": "Trajectory Overlay",
      "imageUrl": "/images/projects/autonomous_warehouse_navigation_with/image02.webp",
      "telemetryNote": "Shows DQN, BFS, and A* all reaching the goal on moderate_map in 22 steps with 0 collisions."
    },
    {
      "title": "Reinforcement Learning Training Diagnostics",
      "badge": "Training Curves",
      "imageUrl": "/images/projects/autonomous_warehouse_navigation_with/image03.webp",
      "telemetryNote": "Tracks cumulative reward convergence, Huber Bellman loss stabilization, 100% success saturation, and epsilon decay."
    },
    {
      "title": "Comparative Baseline Benchmark",
      "badge": "Benchmark Table",
      "imageUrl": "/images/projects/autonomous_warehouse_navigation_with/image04.webp",
      "telemetryNote": "Compares BFS, A*, and DQN across simple, moderate, complex, and generalization maps on path length, collisions, and latency."
    },
    {
      "title": "Zero-Shot Scenario Stress Testing",
      "badge": "Failure Analysis",
      "imageUrl": "/images/projects/autonomous_warehouse_navigation_with/image05.webp",
      "telemetryNote": "Illustrates the 120-step timeout and 100 collisions on generalization_map due to a local minimum obstacle trap."
    },
    {
      "title": "Warehouse Layout & Trajectory Comparison",
      "badge": "Grid & Path Plot",
      "imageUrl": "/images/projects/autonomous_warehouse_navigation_with/image06.webp",
      "telemetryNote": "Contrasts the moderate_map rack layout with collision-free 22-step trajectories from DQN, BFS, and A*."
    },
    {
      "title": "Training Telemetry Summary Grid",
      "badge": "4-Panel Diagnostics",
      "imageUrl": "/images/projects/autonomous_warehouse_navigation_with/image07.webp",
      "telemetryNote": "Captures reward climbing from -261.0 to +66.6, Huber loss convergence, 100% success by episode 126, and step reduction to 22."
    },
    {
      "title": "Multi-Metric Benchmark Evaluation Charts",
      "badge": "4-Panel Benchmark",
      "imageUrl": "/images/projects/autonomous_warehouse_navigation_with/image08.webp",
      "telemetryNote": "Compares success rate, path length, per-step latency (~0.15ms), and collisions across all 4 warehouse scenarios."
    }
  ],
  "mlVisualizationTelemetryNote": "Training converges reliably on moderate_map by episode 126 and transfers zero-shot to simple_map, whereas complex_map and generalization_map expose local-minimum timeouts.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "The DQN agent learns spatial gradients toward the goal while avoiding storage racks via local 3x3 occupancy and 4-way ray-casting sensors. It matches optimal 18-step and 22-step A* paths on open and structured aisle maps in 0.15ms per step, but lacks global memory to escape deep cul-de-sacs.",
    "keyDrivers": [
      {
        "factor": "Potential-Based Manhattan Distance Shaping",
        "impact": "Eliminates sparse-reward wandering and accelerates convergence to optimal 22-step paths within 150 episodes."
      },
      {
        "factor": "4-Way Ray-Casting + 3x3 Local Occupancy Sensors",
        "impact": "Provides immediate aisle clearance and obstacle awareness enabling zero-collision navigation on regular rack layouts."
      },
      {
        "factor": "Soft Polyak Target Updates & Huber Loss",
        "impact": "Stabilizes Bellman target estimation and prevents Q-value divergence during early high-collision exploration."
      }
    ],
    "limitations": [
      {
        "title": "Local Minimum Entrapment in Non-Convex Mazes",
        "description": "On complex_map and generalization_map (0% success, 99-100 collisions), reactive local perception without recurrent memory or global occupancy maps gets trapped behind U-shaped walls."
      },
      {
        "title": "Static Single-Agent Grid Assumptions",
        "description": "The simulator models static storage racks and discrete 4-directional motion without dynamic obstacles, moving forklifts, or continuous differential-drive kinematics."
      },
      {
        "title": "Fixed Training Topology",
        "description": "Training solely on moderate_map limits zero-shot policy robustness when obstacle layouts differ significantly from parallel aisle corridors."
      }
    ],
    "futureImprovements": [
      {
        "title": "Dueling Double DQN with LSTM Memory (DRQN)",
        "target": "Resolve partial observability and escape non-convex dead ends on complex_map and generalization_map."
      },
      {
        "title": "Domain Randomization Across Procedural Layouts",
        "target": "Train across randomized obstacle densities to achieve high zero-shot generalization on unseen floorplans."
      },
      {
        "title": "Multi-Agent & Dynamic Obstacle Extensions",
        "target": "Support moving warehouse agents and continuous velocity control via PPO or SAC."
      }
    ],
    "discoveryNote": "Comparing DQN directly against BFS and A* pinpoints the exact trade-off between O(1) local neural inference and global graph search."
  },
  "mlModelArtifacts": [
    {
      "name": "best_model.pth",
      "format": "PTH"
    },
    {
      "name": "final_model.pth",
      "format": "PTH"
    },
    {
      "name": "training_history.csv",
      "format": "CSV"
    },
    {
      "name": "evaluation_results.csv",
      "format": "CSV"
    },
    {
      "name": "experiment_config.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Reinforcement Learning",
    "modelArchitecture": "Deep Q-Network MLP",
    "coreFramework": "PyTorch",
    "validationMethod": "Deterministic Multi-Map Benchmarking"
  }
}
```
