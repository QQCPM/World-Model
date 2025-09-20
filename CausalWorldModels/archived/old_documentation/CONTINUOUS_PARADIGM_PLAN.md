# CONTINUOUS CAUSAL WORLD MODELS - IMPLEMENTATION PLAN

## 🎯 **PARADIGM SHIFT SUMMARY**

**Previous Approach (FAILED)**: Visual World Models with discrete grid navigation
- Used 64x64 RGB images for simple grid traversal problem
- Applied continuous visual learning to discrete graph problem
- Result: 0% success rate due to fundamental algorithm-problem mismatch

**New Approach (SOLUTION)**: Continuous State World Models
- Direct 12D continuous state vectors (no visual compression needed)
- Physics-based continuous control with momentum and collision
- Causal factors integrated into movement dynamics (not visuals)
- Appropriate algorithm for continuous control problem

---

## 📊 **CONTINUOUS ENVIRONMENT VALIDATION** ✅

**Environment Status**: FULLY FUNCTIONAL
- ✅ **12D state space**: [agent_pos(2), goal_pos(2), velocity(2), goal_distance(1), causal_encoding(5)]
- ✅ **2D continuous control**: velocity commands [-1,1] x [-1,1]
- ✅ **Physics simulation**: PyMunk with momentum, friction, collision
- ✅ **Causal effects**: Weather reduces movement speed (snow: -33%)
- ✅ **Goal reaching**: Confirmed navigation from distance 18.16 → 5.18

---

## 🔄 **REQUIRED SYSTEM UPDATES**

### **IMMEDIATE INCOMPATIBILITIES IDENTIFIED**:
1. **Data Generator**: Uses `TestCampusEnv` (fake) → Must use `ContinuousCampusEnv`
2. **Episode Format**: (100, 64, 64, 3) visual → (episode_length, 12) continuous
3. **Model Architecture**: VAE for visual compression → Direct state prediction models
4. **Training Pipeline**: Image-based → Continuous state sequence modeling

---

## 📋 **IMPLEMENTATION PLAN**

### **PHASE 1: DATA PIPELINE UPDATE** (Priority 1)

#### **Step 1.1: Update Data Generator**
- **File**: `01_generate_causal_data.py`
- **Change**: Replace `TestCampusEnv` with `ContinuousCampusEnv`
- **Output**: Episodes with (episode_length, 12) continuous observations
- **Target**: 200 episodes for initial testing

#### **Step 1.2: Update Exploration Strategy**
- **File**: `utils/exploration.py`
- **Change**: Continuous action space exploration (not discrete)
- **Method**: Gaussian noise on velocity commands with goal-directed bias

#### **Step 1.3: Generate Continuous Episodes**
- **Command**: `python 01_generate_causal_data.py --total_episodes 200 --time_steps 200`
- **Backup**: Move old visual episodes to `data/causal_episodes_backup/`
- **Validation**: Confirm (200, 12) observation shape in new episodes

### **PHASE 2: MODEL ARCHITECTURE UPDATE** (Priority 2)

#### **Step 2.1: Design Continuous State Models**
- **New File**: `continuous_models/state_predictors.py`
- **Architectures**:
  1. **Linear Dynamics**: Simple linear state transition
  2. **LSTM Predictor**: Sequence modeling for temporal dependencies
  3. **GRU Dynamics**: Lightweight recurrent dynamics
  4. **Neural ODE**: Continuous-time dynamics (advanced)
  5. **VAE-RNN Hybrid**: Latent state compression + dynamics

#### **Step 2.2: Update Training Pipeline**
- **File**: `05_train_continuous_models.py` (new)
- **Input**: 12D state sequences
- **Loss**: State prediction MSE + optional causal consistency
- **Output**: Models that predict next_state = f(current_state, action, causal_factors)

### **PHASE 3: NAVIGATION CONTROLLER UPDATE** (Priority 3)

#### **Step 3.1: Continuous Control Policy**
- **File**: `06_continuous_navigation_controller.py` (new)
- **Input**: Current 12D state
- **Output**: 2D velocity command [-1,1] x [-1,1]
- **Method**: Model-based planning with learned dynamics

#### **Step 3.2: Baseline Controllers**
- **Greedy**: Direct movement toward goal
- **Model-Free RL**: Simple Q-learning on continuous actions
- **Model-Based**: Use learned dynamics for planning

### **PHASE 4: EVALUATION PIPELINE** (Priority 4)

#### **Step 4.1: Update Evaluation Scripts**
- **File**: `09_evaluate_continuous_navigation.py` (new)
- **Metrics**: Success rate, path efficiency, causal adaptation
- **Test Episodes**: 50 episodes with varied causal conditions

#### **Step 4.2: Statistical Analysis**
- **File**: `07_statistical_analysis.py` (update)
- **Tests**: Model comparison, significance testing, effect sizes

---

## 🎯 **SUCCESS CRITERIA**

### **Phase 1 Success**: Data Pipeline Working
- ✅ 200 continuous episodes generated successfully
- ✅ Episode format: (episode_length, 12) observations
- ✅ Episodes contain valid causal state variations
- ✅ Navigation episodes show goal-directed movement

### **Phase 2 Success**: Models Training
- ✅ At least 3 model architectures train without errors
- ✅ Prediction accuracy: MSE < 0.1 on validation set
- ✅ Models learn temporal dependencies (t+1 prediction from t)

### **Phase 3 Success**: Navigation Working
- ✅ Model-based controller achieves >20% success rate
- ✅ Beats simple baseline by statistically significant margin
- ✅ Shows adaptation to causal factors (weather, crowds)

### **Phase 4 Success**: Complete System
- ✅ End-to-end pipeline: environment → data → training → evaluation
- ✅ Reproducible results with statistical significance
- ✅ Clear performance comparison across model architectures

---

## ⚡ **IMPLEMENTATION TIMELINE**

### **Day 1** (4-6 hours):
- ✅ Update data generator for continuous environment
- ✅ Generate 200 continuous episodes
- ✅ Validate episode format and content

### **Day 2** (6-8 hours):
- ✅ Implement 3 continuous state prediction models
- ✅ Create training pipeline for 12D sequences
- ✅ Train models and validate convergence

### **Day 3** (4-6 hours):
- ✅ Implement model-based navigation controller
- ✅ Run evaluation on 50 test episodes
- ✅ Statistical analysis and performance comparison

### **Total Estimated**: 14-20 hours for complete continuous system

---

## 🔬 **TECHNICAL SPECIFICATIONS**

### **Continuous State Vector (12D)**:
```python
obs = [
    agent_x,           # Agent position X [0, 100]
    agent_y,           # Agent position Y [0, 100]
    goal_x,            # Goal position X [0, 100]
    goal_y,            # Goal position Y [0, 100]
    velocity_x,        # Agent velocity X [-5, 5]
    velocity_y,        # Agent velocity Y [-5, 5]
    goal_distance,     # Distance to goal [0, 150]
    time_hour_norm,    # Normalized hour [0, 1]
    day_week_norm,     # Normalized day [0, 1]
    weather_effect,    # Weather movement modifier [0.5, 1.0]
    crowd_density,     # Crowd density [0, 1]
    event_modifier     # Event effect on movement [0.5, 1.5]
]
```

### **Model Input/Output**:
```python
# Input: (batch_size, sequence_length, 12 + 2)  # state + action
# Output: (batch_size, sequence_length, 12)     # next state prediction
```

### **Action Space**:
```python
action = [velocity_x, velocity_y]  # Range: [-1, 1] x [-1, 1]
```

---

## 🚀 **READY TO EXECUTE**

**Status**: Plan complete, ready for systematic implementation
**Next Action**: Execute Phase 1, Step 1.1 - Update data generator
**Priority**: Ensure each step works before proceeding to next
**Validation**: Test thoroughly at each phase before advancing

This plan addresses the fundamental paradigm error and provides a systematic path to a working continuous causal world model system.