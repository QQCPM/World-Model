#!/usr/bin/env python3
"""
Enhanced Structure Learning Training
Train structure learner to actually discover causal relationships

Issues to address:
1. Structure learner not finding embedded causal relationships
2. Need stronger training and better initialization
3. Need proper loss weighting
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from causal_architectures import CausalStructureLearner


def train_structure_learner_properly():
    """
    Train structure learner with enhanced methodology to discover relationships
    """
    print("🏗️ ENHANCED STRUCTURE LEARNING TRAINING")
    print("=" * 50)

    # Create structure learner with good parameters
    structure_learner = CausalStructureLearner(
        num_variables=5,
        hidden_dim=32,
        learning_rate=1e-3
    )

    print("1. Generating strong causal data...")

    # Generate data with very clear causal relationships
    batch_size, seq_len = 64, 30  # Larger dataset
    causal_data = torch.zeros(batch_size, seq_len, 5)

    # Initialize with diverse starting conditions
    causal_data[:, 0, :] = torch.randn(batch_size, 5) * 0.5

    # Generate with very strong, clear causal relationships
    for t in range(1, seq_len):
        # VERY STRONG weather -> crowd relationship (0 -> 1)
        causal_data[:, t, 1] = (0.9 * causal_data[:, t-1, 0] +
                               0.1 * torch.randn(batch_size))

        # STRONG time -> road relationship (3 -> 4)
        causal_data[:, t, 4] = (0.85 * causal_data[:, t-1, 3] +
                               0.15 * torch.randn(batch_size))

        # MEDIUM event -> crowd relationship (2 -> 1)
        event_effect = 0.7 * causal_data[:, t, 2]
        causal_data[:, t, 1] += event_effect * 0.3  # Add to existing weather effect

        # Root causes evolve more slowly
        causal_data[:, t, 0] = (0.95 * causal_data[:, t-1, 0] +
                               0.05 * torch.randn(batch_size))  # weather (slow change)
        causal_data[:, t, 3] = (0.95 * causal_data[:, t-1, 3] +
                               0.05 * torch.randn(batch_size))  # time (slow change)

        # Events are more random but persistent
        causal_data[:, t, 2] = (0.7 * causal_data[:, t-1, 2] +
                               0.3 * torch.randn(batch_size))

        # Clamp to prevent extreme values
        causal_data[:, t, :] = torch.clamp(causal_data[:, t, :], -2.0, 2.0)

    print(f"✅ Generated enhanced causal data: {causal_data.shape}")

    # Verify causal relationships in data
    print("2. Verifying causal relationships in generated data...")

    # Check weather -> crowd correlation
    weather = causal_data[:, :-1, 0].flatten()  # t to t+1
    crowd_next = causal_data[:, 1:, 1].flatten()
    weather_crowd_corr = torch.corrcoef(torch.stack([weather, crowd_next]))[0, 1]

    # Check time -> road correlation
    time = causal_data[:, :-1, 3].flatten()
    road_next = causal_data[:, 1:, 4].flatten()
    time_road_corr = torch.corrcoef(torch.stack([time, road_next]))[0, 1]

    print(f"   Weather -> Crowd correlation: {weather_crowd_corr:.3f}")
    print(f"   Time -> Road correlation: {time_road_corr:.3f}")

    if abs(weather_crowd_corr) > 0.5 and abs(time_road_corr) > 0.5:
        print("✅ Strong causal relationships confirmed in data")
    else:
        print("⚠️  Causal relationships may be weak")

    print("3. Training structure learner with enhanced methodology...")

    # Enhanced training
    optimizer = optim.Adam(structure_learner.parameters(), lr=2e-3)  # Higher learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

    # Track training progress
    losses = []
    dag_violations = []

    for epoch in range(50):  # More epochs
        optimizer.zero_grad()

        loss, loss_info = structure_learner.compute_structure_loss(causal_data)

        # Add additional pressure for structure discovery
        adjacency = structure_learner.get_adjacency_matrix()

        # Encourage some edges (but not too many)
        edge_count = torch.sum(torch.abs(adjacency) > 0.1)
        if edge_count < 2:  # Too sparse
            sparsity_penalty = -0.1 * edge_count  # Negative penalty = encouragement
        elif edge_count > 8:  # Too dense
            sparsity_penalty = 0.1 * (edge_count - 8)
        else:
            sparsity_penalty = 0.0

        total_loss = loss + sparsity_penalty

        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(structure_learner.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        dag_violations.append(abs(loss_info['dag_constraint']))

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:2d}: loss={loss.item():.4f}, DAG={loss_info['dag_constraint']:.6f}, edges={edge_count}")

    print("4. Analyzing final learned structure...")

    graph_summary = structure_learner.get_causal_graph_summary()
    final_adjacency = structure_learner.get_adjacency_matrix()

    print(f"✅ Training complete:")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   DAG violation: {dag_violations[-1]:.6f}")
    print(f"   Learned edges: {graph_summary['num_edges']}")
    print(f"   Graph sparsity: {graph_summary['sparsity']:.3f}")

    # Show adjacency matrix
    adj_np = final_adjacency.detach().cpu().numpy()
    print("   Adjacency matrix:")
    var_names = ['weather', 'crowd', 'event', 'time', 'road']
    print("        " + "  ".join([f"{name:>7}" for name in var_names]))
    for i, name in enumerate(var_names):
        row_str = f"{name:>7}: " + "  ".join([f"{adj_np[i,j]:7.3f}" for j in range(5)])
        print(row_str)

    if graph_summary['edges']:
        print("   Discovered relationships:")
        for edge in graph_summary['edges']:
            print(f"     {edge['cause']} → {edge['effect']} (weight: {edge['weight']:.3f})")
    else:
        print("   ⚠️  No significant edges discovered")

    # Check if we found the expected relationships
    expected_relationships = [
        (0, 1),  # weather -> crowd
        (3, 4),  # time -> road
        (2, 1),  # event -> crowd (optional)
    ]

    found_relationships = []
    for edge in graph_summary['edges']:
        cause_idx = edge['cause_idx']
        effect_idx = edge['effect_idx']
        found_relationships.append((cause_idx, effect_idx))

    correct_discoveries = 0
    for expected in expected_relationships:
        if expected in found_relationships:
            correct_discoveries += 1

    print(f"   Structure discovery accuracy: {correct_discoveries}/{len(expected_relationships)} expected relationships found")

    return structure_learner, causal_data, graph_summary


if __name__ == "__main__":
    train_structure_learner_properly()