## Project Plan: Causal World Model for Campus Navigation
1. Ultimate Goal

The goal is to build an agent that understands its environment, not just performs well in it.

Standard Agent: Learns a skill (e.g., how to drive a specific car). Fails when the situation changes.

Our Agent: Acquires knowledge (e.g., the underlying rules of campus life). Can adapt to new situations it has never seen before.

The final test is to see if the agent can reason its way through a novel scenario. For example, can it create a smart plan for a rainy football gameday, even if it has never experienced that exact combination in training?

2. The Core Idea: A Mind in Three Parts

We will build an agent with three distinct modules:

The Perceiver (V): The "eye." Takes a complex image of the campus and compresses it into a simple, abstract thought or concept (z).

The Dream Engine (M): The "imagination." Learns the rules of the world. It predicts what the next concept (z_t+1) will be, given the current concept (z_t), an action, and the causal state of the world (e.g., weather=rain).

The Actor (C): The "decision-maker." A simple brain that learns to achieve goals by "thinking" and practicing entirely within the Dream Engine's imagination.

3. The Key Innovation: Causal Intervention

This is what makes the project novel. We don't just build a world model; we build a causal one.

We define meaningful causal variables relevant to campus life:

time: 8, day: 'Monday'

weather: 'sunny', 'rain'

event: 'normal', 'football_gameday'

The Dream Engine learns how these variables change the rules of the world.

This allows the agent to perform "Interventional Dreams"—to ask counterfactual questions like, "What would this path look like if it were raining right now?" This is the foundation of its reasoning ability.

4. The Research & Development Plan

This is a structured, scientific approach to building the agent. We will run parallel experiments to test hypotheses at each stage.

Phase 1: Architecture Validation (Finding the Best "Eye")

Question: What's the best way to represent the campus visually?

Method: "Bake-off" between different Perceiver architectures (VAE, VQ-VAE, your hierarchical static/dynamic idea, etc.).

Goal: Select the architecture that creates the most useful and structured latent space z.

Phase 2: Causal Factor Analysis (Finding What Matters)

Question: Which causal variables are most important for predicting the future?

Method: Systematically test the Dream Engine's predictive accuracy with different combinations of causal information (time only, weather only, all variables, no variables).

Goal: Prove that the causal framework provides a measurable benefit and understand which factors are most influential.

Phase 3: Replay Strategy Experiments (Finding the Best Way to "Think")

Question: What is the most efficient way for the agent to learn from its dreams?

Method: Test different learning algorithms inside the dream (e.g., prioritizing mistakes, planning forward, credit assignment backward).

Goal: Discover the optimal strategy for turning the world model's knowledge into intelligent behavior.


  Phase 1: Architecture Validation (Weeks 1-2)
  # Run 6 architectures simultaneously (~18GB each)
  experiments = {
      "baseline_32D": Original World Models,
      "gaussian_256D": Bigger Gaussian VAE,
      "categorical_1024D": DreamerV3-style categorical,
      "beta_vae_256D": Disentangled β-VAE,
      "vq_vae_256D": IRIS-style discrete tokens,
      "hierarchical_512D": Your static/dynamic separation
  }

  # THIS is proper use of your hardware

  Phase 2: Causal Factor Analysis (Weeks 3-4)
  # Don't test all 32 combinations - test key hypotheses
  causal_experiments = {
      "temporal_only": [time, day],
      "environmental_only": [weather],
      "social_only": [crowd, event],
      "full_causal": [time, day, weather, crowd, event],
      "static_dynamic": Your factorization approach,
      "no_causal": Baseline
  }

  # 6 models running simultaneously
  # Question: Which factors are actually learnable?

  Phase 3: Replay Strategy Experiments (Weeks 5-6)
  # THIS is where neuroscience becomes actionable
  replay_strategies = {
      "uniform": Random experience replay,
      "prioritized": High-error episodes,
      "forward_sweep": Plan future trajectories,
      "reverse_sweep": Credit assignment from rewards,
      "wake_sleep": Online vs offline learning modes
  }

  # Question: Does biological replay improve world models?