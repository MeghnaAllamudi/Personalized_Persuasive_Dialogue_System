import sys
sys.path.append('.')
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from models.strategy_selector import StrategySelector

sns.set_style("whitegrid")

# Figure 1: Few-shot adaptation speed
with open('results/few_shot_adaptation_results_fast.json') as f:
    adaptation_results = json.load(f)

# Auto-detect which baselines are available
available_baselines = list(adaptation_results.keys())
print(f"Detected baselines: {available_baselines}")

# Auto-detect shot sizes
available_shots = sorted([int(k) for k in adaptation_results['maml'].keys()])
num_shots = available_shots  # Use all available shot sizes

fig, ax = plt.subplots(figsize=(10, 6))

# Build methods dict based on what's available
methods = {
    'maml': ('MAML (Ours)', 'steelblue', 'o'),
    'baseline_random': ('Random Selection', 'coral', 's')
}

if 'baseline_vanilla' in available_baselines:
    methods['baseline_vanilla'] = ('Vanilla LLM', 'gray', 'x')
    print("Using Vanilla LLM baseline")
if 'baseline_population' in available_baselines:
    methods['baseline_population'] = ('Population-Best', 'lightgreen', '^')
    print("Using Population-Best baseline")
if 'baseline_oracle' in available_baselines:
    methods['baseline_oracle'] = ('Oracle (Upper Bound)', 'gold', 'D')
    print("Using Oracle baseline")
if 'baseline_scratch' in available_baselines and 'baseline_population' not in available_baselines:
    methods['baseline_scratch'] = ('Train from Scratch', 'lightgreen', '^')
    print("Using Scratch baseline")

for method_key, (label, color, marker) in methods.items():
    means = [np.mean(adaptation_results[method_key][str(k)]) for k in num_shots]
    stds = [np.std(adaptation_results[method_key][str(k)]) for k in num_shots]
    
    ax.plot(num_shots, means, marker=marker, label=label, color=color, 
            linewidth=2.5, markersize=10, alpha=0.8)
    ax.fill_between(num_shots, 
                     np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds),
                     alpha=0.2, color=color)

ax.set_xlabel('Number of Adaptation Examples (K)', fontsize=13)
ax.set_ylabel('Average Reward on Test Episodes', fontsize=13)
ax.set_title('MAML Enables Rapid Adaptation to New Personas', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xticks(num_shots)

plt.tight_layout()
plt.savefig('results/maml_adaptation_speed.png', dpi=300, bbox_inches='tight')
print("Saved: results/maml_adaptation_speed.png")

# Figure 2: Strategy personalization heatmap
# Load MAML model and test strategy preferences for each persona
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = StrategySelector(state_dim=384, num_strategies=5)
model.load_state_dict(torch.load('results/maml_model.pt', map_location=device))
model.to(device)
model.eval()

from models.maml_trainer import MAMLTrainer
from models.state_encoder import ConversationStateEncoder
from models.persona_simulator import CasinoPersonaSimulator

trainer = MAMLTrainer(model, inner_lr=0.01, device=device)
encoder = ConversationStateEncoder()
strategies = ['Empathy', 'Validation', 'Active\nListening', 'Problem\nSolving', 'Authority']

persona_types = ['aggressive', 'cooperative', 'anxious', 'stubborn', 'diplomatic']
persona_labels = ['Aggressive', 'Cooperative', 'Anxious', 'Stubborn', 'Diplomatic']

strategy_preferences = np.zeros((len(persona_types), len(strategies)))

for p_idx, persona_type in enumerate(persona_types):
    # Generate support data
    support_data = []
    simulator = CasinoPersonaSimulator(persona_type, model="gpt-4o-mini")
    
    # Collect 3 turns
    from models.strategy_generator import StrategyPromptedGenerator
    generator = StrategyPromptedGenerator(model="gpt-4o-mini")
    conversation = []
    
    for turn in range(3):
        strat = [strategies[turn % len(strategies)].replace('\n', ' ')]
        agent_msg = generator.generate_response(conversation, strat)
        user_msg = simulator.get_response(agent_msg, strat)
        
        state = encoder.encode_conversation(conversation)
        action = np.zeros(5)
        action[turn % 5] = 1
        reward = np.random.randn()  # Dummy
        
        support_data.append({'state': state, 'action': action, 'reward': reward})
        conversation.append({'agent': agent_msg, 'user': user_msg})
    
    # Adapt model
    adapted_params = trainer._inner_loop(support_data, create_graph=False)
    
    # Get strategy preferences
    test_state = encoder.encode_conversation(conversation)
    probs = trainer.predict_with_params(adapted_params, test_state)
    strategy_preferences[p_idx] = probs

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(strategy_preferences, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=strategies, yticklabels=persona_labels,
            cbar_kws={'label': 'Selection Probability'}, ax=ax,
            vmin=0, vmax=1)

ax.set_title('MAML Learns Persona-Specific Strategy Preferences', fontsize=15, fontweight='bold')
ax.set_xlabel('Strategy', fontsize=13)
ax.set_ylabel('Persona Type', fontsize=13)

plt.tight_layout()
plt.savefig('results/persona_strategy_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: results/persona_strategy_heatmap.png")

plt.show()
