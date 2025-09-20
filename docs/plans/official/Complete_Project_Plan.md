# Causal Campus Navigation World Models
## Complete Implementation Plan

### Project Overview

**Objective**: Build a modern World Model with explicit causal reasoning for campus navigation, extending the 2018 World Models paper with 2024-2025 techniques and novel causal conditioning.

**Core Innovation**: Instead of learning `p(z_t+1 | z_t, action)`, learn `p(z_t+1 | z_t, action, causal_state)` where causal_state includes temporal, environmental, and social factors that affect navigation dynamics.

**Hardware**: M2 Ultra 128GB RAM - optimized for massive parallel experimentation

---

## Technical Architecture

### Base Architecture (Modern World Models 2024)
```python
# Data Flow: 
campus_obs(64x64x3) + causal_state(45D) → 
Categorical_VAE → z(256D/1024D) → 
Causal_GRU → action(5) + value_prediction
```

### Key Modernizations from 2018
- **VAE**: Categorical latents (32×32) instead of Gaussian (32D)
- **RNN**: GRU with layer normalization instead of LSTM
- **Training**: Symlog transform, two-hot encoding, free bits
- **Innovation**: Causal conditioning + static/dynamic factorization

---

## Environment: Simple Campus (Option 1)

### Campus Layout
```python
class SimpleCampusEnv:
    def __init__(self):
        # 64x64 grid world representation
        self.buildings = {
            'library': (15, 15, 25, 25),      # Academic building
            'gym': (40, 10, 50, 20),          # Recreation center  
            'cafeteria': (10, 40, 20, 50),    # Dining facility
            'academic': (35, 35, 50, 50)      # Classroom building
        }
        self.paths = self.generate_path_network()
        self.spawn_points = [(5,5), (30,30), (55,55)]
```

### Causal Variables (45 dimensions total)
```python
causal_state = {
    'time_hour': 24,        # 0-23 (one-hot)
    'day_week': 7,          # Monday-Sunday (one-hot)
    'weather': 4,           # sunny/rain/snow/fog (one-hot)
    'event': 5,             # normal/gameday/exam/break/construction (one-hot)
    'crowd_density': 5      # very_low to very_high (one-hot)
}
# Total: 24+7+4+5+5 = 45 dimensions
```

### Visual Causal Effects
- **Rain**: Darker overall appearance + puddle sprites
- **Gameday**: Dense red crowd pixels near certain buildings
- **Night**: 50% brightness reduction + different lighting
- **Construction**: Orange barrier sprites blocking paths
- **Exam Week**: Higher crowd density near library

---

## Phase 1: Architecture Validation (Weeks 1-3)

### Experiments (8 parallel models)

```python
architecture_experiments = {
    # Core Architectures
    "baseline_32D": {
        "vae_type": "gaussian",
        "latent_dim": 32,
        "architecture": "original_world_models"
    },
    
    "gaussian_256D": {
        "vae_type": "gaussian", 
        "latent_dim": 256,
        "architecture": "modern_conv_blocks"
    },
    
    "categorical_1024D": {
        "vae_type": "categorical",
        "num_categoricals": 32,
        "num_classes": 32,  # Total: 32×32 = 1024D
        "architecture": "dreamerv3_style"
    },
    
    "beta_vae_4.0": {
        "vae_type": "beta_vae",
        "latent_dim": 256,
        "beta": 4.0,  # Moderate disentanglement
        "architecture": "modern_conv_blocks"
    },
    
    "vq_vae_256D": {
        "vae_type": "vq_vae",
        "codebook_size": 256,
        "commitment_cost": 0.25,
        "architecture": "iris_style"
    },
    
    "hierarchical_512D": {
        "vae_type": "hierarchical",
        "static_dim": 256,      # Building layouts
        "dynamic_dim": 256,     # Crowds, weather effects
        "architecture": "our_innovation"
    },
    
    # Ablation Studies  
    "no_conv_normalization": {
        "vae_type": "gaussian",
        "latent_dim": 256,
        "layer_norm": False,  # Test importance of normalization
        "architecture": "ablation"
    },
    
    "deeper_encoder": {
        "vae_type": "gaussian",
        "latent_dim": 256,
        "encoder_layers": 8,    # vs standard 4
        "architecture": "depth_test"
    }
}
```

### Phase 1 Success Metrics
- **Reconstruction quality**: SSIM > 0.85 on campus images
- **Latent utilization**: Active dimensions > 80% of total
- **Training stability**: Loss convergence within 50 epochs
- **Resource efficiency**: Memory usage < 12GB per model

### Memory Allocation Phase 1
```
8 models × 12GB each = 96GB RAM usage
Remaining: 32GB for system + data loading
Training time: 2-3 days parallel
```

---

## Phase 2A: Core Causal Validation (Weeks 4-6)

### Experiments (6 parallel models)

```python
causal_core_experiments = {
    "no_causal": {
        "causal_factors": [],
        "description": "Baseline - no causal conditioning",
        "input_dim": "latent_dim + action_dim"  # No causal vars
    },
    
    "temporal_only": {
        "causal_factors": ["time_hour", "day_week"],
        "description": "Time-based effects only",
        "input_dim": "latent_dim + action_dim + 31"  # 24+7
    },
    
    "environmental_only": {
        "causal_factors": ["weather"],
        "description": "Weather effects only", 
        "input_dim": "latent_dim + action_dim + 4"
    },
    
    "social_only": {
        "causal_factors": ["event", "crowd_density"],
        "description": "Social/event effects only",
        "input_dim": "latent_dim + action_dim + 10"  # 5+5
    },
    
    "temporal_environmental": {
        "causal_factors": ["time_hour", "day_week", "weather"],
        "description": "Test factor interaction",
        "input_dim": "latent_dim + action_dim + 35"  # 24+7+4
    },
    
    "full_causal": {
        "causal_factors": ["time_hour", "day_week", "weather", "event", "crowd_density"],
        "description": "All causal factors",
        "input_dim": "latent_dim + action_dim + 45"
    }
}
```

### Causal GRU Architecture
```python
class CausalMDNGRU(nn.Module):
    def __init__(self, latent_dim, action_dim, causal_dim):
        super().__init__()
        
        # Modern GRU with layer normalization
        self.gru = nn.GRU(
            input_size=latent_dim + action_dim + causal_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer normalization (DreamerV3 trick)
        self.layer_norm = nn.LayerNorm(512)
        
        # Prediction heads
        self.latent_head = GaussianMixtureHead(512, latent_dim, num_mixtures=5)
        self.reward_head = TwoHotHead(512, num_bins=255, min_val=-20, max_val=20)
        self.value_head = TwoHotHead(512, num_bins=255, min_val=-20, max_val=20)
        
        # Modern training techniques
        self.use_symlog = True
        self.free_bits = 1.0
        
    def forward(self, z, action, causal_state, hidden=None):
        # Concatenate all inputs
        inputs = torch.cat([z, action, causal_state], dim=-1)
        
        # GRU forward pass
        gru_out, new_hidden = self.gru(inputs, hidden)
        gru_out = self.layer_norm(gru_out)
        
        # Predictions
        next_z_dist = self.latent_head(gru_out)
        reward_dist = self.reward_head(gru_out)
        value_dist = self.value_head(gru_out)
        
        return next_z_dist, reward_dist, value_dist, new_hidden
```

### Phase 2A Success Metrics
- **Causal sensitivity**: Different predictions for different causal states
- **Intervention accuracy**: Model correctly predicts "what if weather changed?"
- **Generalization**: Performance on held-out causal combinations
- **Factor importance**: Which causal factors matter most for navigation?

### Memory Allocation Phase 2A
```
6 models × 10GB each = 60GB RAM usage  
Remaining: 68GB for data + multiple environment instances
Training time: 2-3 days parallel
```

---

## Training Pipeline Modifications

### Data Generation (01_generate_causal_data.py)
```python
class CausalDataGenerator:
    def __init__(self, num_parallel_envs=32):
        self.environments = [SimpleCampusEnv() for _ in range(num_parallel_envs)]
        self.causal_scheduler = CausalStateScheduler()
        
    def generate_structured_episodes(self):
        # Systematic causal coverage instead of random exploration
        causal_combinations = self.get_structured_combinations()
        
        # Parallel episode collection
        with mp.Pool(32) as pool:
            episodes = pool.map(self.collect_episode_with_causals, 
                              causal_combinations)
        return episodes
        
    def get_structured_combinations(self):
        # 50 key causal combinations covering:
        # - All individual factors
        # - Important interactions  
        # - Edge cases (night + rain + gameday)
        return structured_combinations
```

### VAE Training (02_train_hierarchical_vae.py)
```python
class ModernVAETrainer:
    def __init__(self, vae_config):
        self.vae = self.build_vae(vae_config)
        self.use_categorical = vae_config.get('categorical', False)
        self.use_hierarchical = vae_config.get('hierarchical', False)
        
    def train_step(self, batch):
        # Modern VAE loss with DreamerV3 improvements
        recon_loss = self.reconstruction_loss(batch)
        
        if self.use_categorical:
            kl_loss = self.categorical_kl_loss(batch)
        else:
            kl_loss = self.gaussian_kl_loss(batch)
            
        # Free bits technique (DreamerV3)
        kl_loss = torch.clamp(kl_loss, min=self.free_bits)
        
        total_loss = recon_loss + kl_loss
        return total_loss
```

### RNN Training (04_train_causal_rnn.py)
```python
class CausalRNNTrainer:
    def __init__(self, rnn_config):
        self.rnn = CausalMDNGRU(**rnn_config)
        self.causal_factors = rnn_config['causal_factors']
        
    def prepare_batch(self, episodes):
        # Extract latent sequences from VAE
        # Add causal state sequences
        # Return (z_seq, action_seq, causal_seq, target_seq)
        
    def loss_function(self, predictions, targets):
        # Modern mixture density loss with symlog transform
        next_z_pred, reward_pred, value_pred = predictions
        next_z_target, reward_target, value_target = targets
        
        # Apply symlog to rewards/values
        reward_target = self.symlog(reward_target)
        value_target = self.symlog(value_target)
        
        # Mixture density loss for latent prediction
        z_loss = self.mdn_loss(next_z_pred, next_z_target)
        
        # Two-hot cross-entropy for rewards/values
        reward_loss = self.twohot_loss(reward_pred, reward_target)
        value_loss = self.twohot_loss(value_pred, value_target)
        
        return z_loss + reward_loss + value_loss
```

---

## Validation Experiments

### Causal Intervention Testing
```python
def test_causal_interventions():
    """Test if model understands causal relationships"""
    
    # Scenario 1: Weather intervention
    # "What happens if it suddenly starts raining?"
    base_state = {'time': 14, 'weather': 'sunny', 'event': 'normal'}
    intervention_state = {'time': 14, 'weather': 'rain', 'event': 'normal'}
    
    # Model should predict: slower movement, different path choices
    
    # Scenario 2: Event intervention  
    # "What happens if gameday is announced?"
    base_state = {'time': 14, 'weather': 'sunny', 'event': 'normal'}
    intervention_state = {'time': 14, 'weather': 'sunny', 'event': 'gameday'}
    
    # Model should predict: heavy crowds, certain areas inaccessible
    
def test_held_out_combinations():
    """Test generalization to unseen causal combinations"""
    
    # Train on individual factors and pairs
    # Test on: night + rain + gameday (never seen together)
    # Success = reasonable navigation despite novel combination
```

### Static/Dynamic Factorization Analysis
```python
def analyze_factorization():
    """Test if model learns static vs dynamic separation"""
    
    # Static factors (should be invariant):
    # - Building locations across weather changes
    # - Path connectivity across time changes
    
    # Dynamic factors (should change):
    # - Crowd density with time/events
    # - Obstacle appearance with weather
    
    # Measure: correlation between latent changes and causal changes
```

---

## Timeline and Milestones

### Week 1: Environment & Infrastructure
- ✅ Clone and setup World Models repository
- ✅ Implement SimpleCampusEnv with causal rendering
- ✅ Setup parallel training infrastructure
- ✅ Test data generation pipeline

### Week 2-3: Phase 1 Architecture Experiments
- ✅ Launch 8 parallel architecture experiments  
- ✅ Monitor training progress and resource usage
- ✅ Analyze reconstruction quality and latent utilization
- ✅ Select best 2-3 architectures for Phase 2A

### Week 4: Phase 2A Setup
- ✅ Implement CausalMDNGRU architecture
- ✅ Modify training pipeline for causal conditioning
- ✅ Generate structured causal dataset
- ✅ Setup causal intervention testing framework

### Week 5-6: Phase 2A Causal Experiments  
- ✅ Launch 6 parallel causal experiments
- ✅ Monitor causal sensitivity and learning dynamics
- ✅ Run intervention testing on trained models
- ✅ Analyze which causal factors matter most

### Week 7: Analysis and Documentation
- ✅ Compare all experimental results
- ✅ Generate visualizations and performance metrics
- ✅ Document key findings and insights
- ✅ Plan potential Phase 2B experiments

---

## Resource Requirements

### Computational Resources
- **Total RAM Usage**: 96GB (Phase 1) + 60GB (Phase 2A) = 156GB peak
- **Strategy**: Run phases sequentially to stay within 128GB limit
- **Training Time**: ~6 days total (3 days per phase)
- **Storage**: ~500GB for datasets, models, and logs

### Memory Optimization Strategies
```python
# Gradient accumulation to reduce batch memory
BATCH_SIZE = 32  # Smaller batches
ACCUMULATION_STEPS = 4  # Simulate larger batches

# Mixed precision training
torch.cuda.amp.autocast(enabled=True)

# Model checkpointing to trade compute for memory
torch.utils.checkpoint.checkpoint(model_layer, inputs)
```

---

## Expected Outcomes

### Research Contributions
1. **Empirical Analysis**: Which VAE architectures work best for spatial reasoning?
2. **Causal Factors**: Which environmental factors are learnable vs. random noise?
3. **Factor Interactions**: Do temporal + environmental effects show synergy?
4. **Architectural Innovations**: Does static/dynamic factorization emerge naturally?

### Practical Applications
1. **Campus Navigation**: Real-world applicable navigation system
2. **Causal Reasoning**: Framework for incorporating domain knowledge into world models
3. **Scalable Training**: Methodology for parallel world model experimentation

### Learning Outcomes
1. **Modern ML Practices**: Experience with 2024 techniques (categorical VAEs, symlog, etc.)
2. **Experimental Design**: Systematic approach to architecture search
3. **Causal Modeling**: Practical experience with intervention testing
4. **High-Performance Computing**: Optimal use of available hardware resources

---

## Risk Mitigation

### Technical Risks
- **Memory Overflow**: Phase sequential execution, gradient accumulation
- **Training Instability**: Multiple checkpoint saves, learning rate scheduling
- **Poor Convergence**: Ablation studies to isolate problematic components

### Timeline Risks  
- **Experiment Delays**: Prioritize core experiments, defer nice-to-have analyses
- **Analysis Complexity**: Prepare automated evaluation scripts in advance
- **Hardware Issues**: Cloud backup plan for computational resources

### Research Risks
- **Weak Causal Effects**: Start with obvious causal factors (weather → movement speed)
- **No Factorization**: Explicit regularization terms to encourage static/dynamic separation
- **Poor Generalization**: Systematic held-out testing from day one

---

## Success Criteria

### Phase 1 Success
- ✅ At least 2 architectures achieve stable training
- ✅ Clear performance differences between architectures
- ✅ Resource utilization < 100GB RAM  
- ✅ Reconstruction quality sufficient for navigation

### Phase 2A Success
- ✅ Causal models outperform no-causal baseline
- ✅ Intervention testing shows expected behavioral changes
- ✅ At least 2 causal factors show clear learning signal
- ✅ Models generalize to held-out causal combinations

### Overall Project Success
- ✅ Novel contribution to world models literature
- ✅ Systematic empirical analysis of architectural choices  
- ✅ Working demonstration of causal reasoning in navigation
- ✅ Methodology for parallel world model experimentation

---

*This plan balances ambitious research goals with practical constraints, using your M2 Ultra's capabilities optimally while maintaining focus on the core innovation: causal reasoning in world models.*