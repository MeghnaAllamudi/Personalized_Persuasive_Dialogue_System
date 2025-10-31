"""
Compare MAML vs Baseline Performance
Shows that persona-specific adaptation leads to better outcomes
"""
import sys
sys.path.append('.')
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
with open('results/few_shot_adaptation_results_fast.json') as f:
    results = json.load(f)

sns.set_style("whitegrid")

# Auto-detect which baselines are available (old vs new format)
available_methods = list(results.keys())
print(f"Detected baselines: {available_methods}")

# Determine which baseline format we have
if 'baseline_population' in available_methods:
    # New format with experiment 06 baselines
    methods = ['maml', 'baseline_random', 'baseline_population', 'baseline_oracle']
    method_labels = ['MAML\n(Adapted)', 'Random', 'Population\nBest', 'Oracle\n(Upper Bound)']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    use_new_baselines = True
    print("Using new baseline format (experiment 06)")
elif 'baseline_scratch' in available_methods:
    # Old format with scratch baseline
    methods = ['maml', 'baseline_random', 'baseline_scratch']
    method_labels = ['MAML\n(Adapted)', 'Random', 'Scratch\n(Overfit)']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    use_new_baselines = False
    print("Using old baseline format (scratch)")
else:
    raise ValueError(f"Unexpected baseline format. Found: {available_methods}")

# Create comprehensive comparison figure
fig = plt.figure(figsize=(16, 10))

# ========== SUBPLOT 1: Performance Comparison by Shot Size ==========
ax1 = plt.subplot(2, 3, 1)

# Auto-detect shot sizes from results
shot_sizes = sorted([int(k) for k in results['maml'].keys()])
print(f"Testing shot sizes: {shot_sizes}")

x = np.arange(len(shot_sizes))
width = 0.8 / len(methods)  # Dynamic width based on number of methods

for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
    means = [np.mean(results[method][str(k)]) for k in shot_sizes]
    stds = [np.std(results[method][str(k)]) for k in shot_sizes]
    
    ax1.bar(x + i*width, means, width, label=label, color=color, alpha=0.8)
    ax1.errorbar(x + i*width, means, yerr=stds, fmt='none', color='black', capsize=3, alpha=0.5)

ax1.set_xlabel('Number of Adaptation Examples (K-shot)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
baseline_type = "New Baselines (Exp 06)" if use_new_baselines else "Old Baselines (Scratch)"
ax1.set_title(f'MAML vs {baseline_type}: Few-Shot Adaptation', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * (len(methods) - 1) / 2)  # Center between bars
ax1.set_xticklabels([f'{k}-shot' for k in shot_sizes])
ax1.legend(fontsize=9, loc='best')
ax1.grid(axis='y', alpha=0.3)

# ========== SUBPLOT 2: Improvement Over Random Baseline ==========
ax2 = plt.subplot(2, 3, 2)

improvements = []
for k in shot_sizes:
    maml_mean = np.mean(results['maml'][str(k)])
    random_mean = np.mean(results['baseline_random'][str(k)])
    improvement = ((maml_mean - random_mean) / abs(random_mean)) * 100 if random_mean != 0 else 0
    improvements.append(improvement)

bars = ax2.bar(shot_sizes, improvements, color=['#2E86AB' if x > 0 else '#A23B72' for x in improvements], alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.axhline(y=10, color='green', linestyle='--', linewidth=1, alpha=0.5, label='POC Success (>10%)')
ax2.set_xlabel('K-shot', fontsize=12, fontweight='bold')
ax2.set_ylabel('Improvement over Random (%)', fontsize=12, fontweight='bold')
ax2.set_title('MAML Advantage: Personalization Pays Off', fontsize=14, fontweight='bold')
ax2.set_xticks(shot_sizes)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (k, imp) in enumerate(zip(shot_sizes, improvements)):
    ax2.text(k, imp + 1, f'{imp:+.1f}%', ha='center', fontweight='bold', fontsize=10)

# ========== SUBPLOT 3: Win Rate Analysis ==========
ax3 = plt.subplot(2, 3, 3)

win_rates = {}
total_episodes = {}

for k in shot_sizes:
    maml_rewards = results['maml'][str(k)]
    random_rewards = results['baseline_random'][str(k)]
    
    label = f'{k}-shot'
    wins = sum(1 for m, r in zip(maml_rewards, random_rewards) if m > r)
    total = len(maml_rewards)
    
    win_rates[label] = (wins / total) * 100
    total_episodes[label] = total

labels = list(win_rates.keys())
values = list(win_rates.values())
bars = ax3.bar(labels, values, color='#2E86AB', alpha=0.8)
ax3.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Random Chance (50%)')
ax3.set_ylabel('Win Rate vs Random (%)', fontsize=12, fontweight='bold')
ax3.set_title('MAML Wins More Episodes', fontsize=14, fontweight='bold')
ax3.set_ylim([0, 100])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for i, (label, value) in enumerate(zip(labels, values)):
    ax3.text(i, value + 2, f'{value:.1f}%', ha='center', fontweight='bold', fontsize=10)

# ========== SUBPLOT 4: Distribution Comparison (1-shot) ==========
ax4 = plt.subplot(2, 3, 4)

maml_1shot = results['maml']['1']
random_1shot = results['baseline_random']['1']

ax4.hist(random_1shot, bins=15, alpha=0.6, color='#A23B72', label='Random (No Adaptation)', density=True)
ax4.hist(maml_1shot, bins=15, alpha=0.6, color='#2E86AB', label='MAML (Adapted)', density=True)
ax4.axvline(np.mean(random_1shot), color='#A23B72', linestyle='--', linewidth=2, label=f'Random Mean: {np.mean(random_1shot):.2f}')
ax4.axvline(np.mean(maml_1shot), color='#2E86AB', linestyle='--', linewidth=2, label=f'MAML Mean: {np.mean(maml_1shot):.2f}')
ax4.set_xlabel('Reward', fontsize=12, fontweight='bold')
ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
ax4.set_title('1-Shot: MAML Shifts Distribution Higher', fontsize=14, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# ========== SUBPLOT 5: Distribution Comparison (3-shot) ==========
ax5 = plt.subplot(2, 3, 5)

maml_3shot = results['maml']['3']
random_3shot = results['baseline_random']['3']

ax5.hist(random_3shot, bins=15, alpha=0.6, color='#A23B72', label='Random', density=True)
ax5.hist(maml_3shot, bins=15, alpha=0.6, color='#2E86AB', label='MAML', density=True)
ax5.axvline(np.mean(random_3shot), color='#A23B72', linestyle='--', linewidth=2, label=f'Random: {np.mean(random_3shot):.2f}')
ax5.axvline(np.mean(maml_3shot), color='#2E86AB', linestyle='--', linewidth=2, label=f'MAML: {np.mean(maml_3shot):.2f}')
ax5.set_xlabel('Reward', fontsize=12, fontweight='bold')
ax5.set_ylabel('Density', fontsize=12, fontweight='bold')
ax5.set_title('3-Shot: MAML Maintains Advantage', fontsize=14, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

# ========== SUBPLOT 6: Summary Statistics Table ==========
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

# Create summary table
summary_data = []
for k in shot_sizes:
    maml_mean = np.mean(results['maml'][str(k)])
    random_mean = np.mean(results['baseline_random'][str(k)])
    
    if use_new_baselines:
        population_mean = np.mean(results['baseline_population'][str(k)])
        comparison_mean = population_mean
        comparison_label = 'Pop-Best'
    else:
        scratch_mean = np.mean(results['baseline_scratch'][str(k)])
        comparison_mean = scratch_mean
        comparison_label = 'Scratch'
    
    improvement_random = ((maml_mean - random_mean) / abs(random_mean)) * 100 if random_mean != 0 else 0
    improvement_comparison = ((maml_mean - comparison_mean) / abs(comparison_mean)) * 100 if comparison_mean != 0 else 0
    
    summary_data.append([
        f'{k}-shot',
        f'{maml_mean:.2f}',
        f'{comparison_mean:.2f}',
        f'{improvement_comparison:+.1f}%',
        'WIN' if improvement_comparison > 5 else 'LOSE'
    ])

table = ax6.table(cellText=summary_data,
                  colLabels=['K-shot', 'MAML', comparison_label, 'Improve', 'Result'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.15, 0.2, 0.2, 0.2, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Color code the header
for i in range(5):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code win/loss
for i in range(1, len(summary_data) + 1):
    if summary_data[i-1][4] == 'WIN':
        table[(i, 4)].set_facecolor('#90EE90')
        table[(i, 4)].set_text_props(weight='bold')
    else:
        table[(i, 4)].set_facecolor('#FFB6C1')

ax6.set_title('Summary: MAML Performance', fontsize=14, fontweight='bold', pad=20)

# ========== OVERALL TITLE ==========
fig.suptitle('MAML POC Results: Persona-Specific Adaptation Improves Outcomes', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results/maml_vs_baseline_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/maml_vs_baseline_comparison.png")

# ========== PRINT TEXT SUMMARY ==========
print("\n" + "="*80)
print("MAML VS BASELINE: PERFORMANCE COMPARISON")
print("="*80)

for k in shot_sizes:
    print(f"\n{k}-SHOT LEARNING:")
    print("-" * 60)
    
    maml_mean = np.mean(results['maml'][str(k)])
    maml_std = np.std(results['maml'][str(k)])
    random_mean = np.mean(results['baseline_random'][str(k)])
    random_std = np.std(results['baseline_random'][str(k)])
    
    print(f"  MAML (Adapted):          {maml_mean:.3f} +/- {maml_std:.3f}")
    print(f"  Random:                  {random_mean:.3f} +/- {random_std:.3f}")
    
    improvement_random = ((maml_mean - random_mean) / abs(random_mean)) * 100 if random_mean != 0 else 0
    
    if use_new_baselines:
        population_mean = np.mean(results['baseline_population'][str(k)])
        population_std = np.std(results['baseline_population'][str(k)])
        oracle_mean = np.mean(results['baseline_oracle'][str(k)])
        oracle_std = np.std(results['baseline_oracle'][str(k)])
        
        print(f"  Population-Best:         {population_mean:.3f} +/- {population_std:.3f}")
        print(f"  Oracle (Upper Bound):    {oracle_mean:.3f} +/- {oracle_std:.3f}")
        
        improvement_pop = ((maml_mean - population_mean) / abs(population_mean)) * 100 if population_mean != 0 else 0
        gap_to_oracle = ((oracle_mean - maml_mean) / abs(oracle_mean)) * 100 if oracle_mean != 0 else 0
        
        print(f"\n  MAML Improvement:")
        print(f"     vs Random:         {improvement_random:+.1f}% {'[BETTER]' if improvement_random > 0 else '[WORSE]'}")
        print(f"     vs Population-Best:{improvement_pop:+.1f}% {'[BETTER]' if improvement_pop > 0 else '[WORSE]'}")
        print(f"     Gap to Oracle:     {gap_to_oracle:.1f}% (lower is better)")
        
        # Statistical significance
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(results['maml'][str(k)], results['baseline_population'][str(k)])
        print(f"     p-value vs Pop:    {p_value:.4f} {'[SIGNIFICANT]' if p_value < 0.05 else '[NOT SIGNIFICANT]'}")
    else:
        scratch_mean = np.mean(results['baseline_scratch'][str(k)])
        scratch_std = np.std(results['baseline_scratch'][str(k)])
        
        print(f"  Scratch:                 {scratch_mean:.3f} +/- {scratch_std:.3f}")
        
        improvement_scratch = ((maml_mean - scratch_mean) / abs(scratch_mean)) * 100 if scratch_mean != 0 else 0
        
        print(f"\n  MAML Improvement:")
        print(f"     vs Random:   {improvement_random:+.1f}% {'[BETTER]' if improvement_random > 0 else '[WORSE]'}")
        print(f"     vs Scratch:  {improvement_scratch:+.1f}% {'[BETTER]' if improvement_scratch > 0 else '[WORSE]'}")
        
        # Statistical significance
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(results['maml'][str(k)], results['baseline_scratch'][str(k)])
        print(f"     p-value vs Scratch: {p_value:.4f} {'[SIGNIFICANT]' if p_value < 0.05 else '[NOT SIGNIFICANT]'}")

print("\n" + "="*80)
print("POC CONCLUSION")
print("="*80)

avg_improvement_vs_random = np.mean([
    ((np.mean(results['maml'][str(k)]) - np.mean(results['baseline_random'][str(k)])) / 
     abs(np.mean(results['baseline_random'][str(k)]))) * 100
    for k in shot_sizes
])

print(f"Average improvement vs Random: {avg_improvement_vs_random:+.1f}%")

if use_new_baselines:
    avg_improvement_vs_pop = np.mean([
        ((np.mean(results['maml'][str(k)]) - np.mean(results['baseline_population'][str(k)])) / 
         abs(np.mean(results['baseline_population'][str(k)]))) * 100
        for k in shot_sizes
    ])
    print(f"Average improvement vs Population-Best: {avg_improvement_vs_pop:+.1f}%")
    
    if avg_improvement_vs_pop > 5:
        print("\n[SUCCESS] POC PASSED: MAML beats population-best baseline")
        print(f"   Improvement: {avg_improvement_vs_pop:.1f}%")
        print("\nWhy MAML Wins:")
        print("   - Learns persona-specific strategy preferences (see heatmap)")
        print("   - Adapts with just 1-9 examples")
        print("   - Outperforms static baselines (random, population-best)")
        print("   - Approaches oracle performance (knows persona)")
    elif avg_improvement_vs_random > 10:
        print("\n[PARTIAL SUCCESS] MAML beats random but close to population-best")
        print(f"   vs Random: {avg_improvement_vs_random:.1f}%")
        print(f"   vs Pop-Best: {avg_improvement_vs_pop:.1f}%")
    else:
        print("\n[WEAK] POC INCONCLUSIVE: MAML does not clearly outperform baselines")
        print(f"   vs Random: {avg_improvement_vs_random:.1f}%")
        print(f"   vs Pop-Best: {avg_improvement_vs_pop:.1f}%")
else:
    avg_improvement_vs_scratch = np.mean([
        ((np.mean(results['maml'][str(k)]) - np.mean(results['baseline_scratch'][str(k)])) / 
         abs(np.mean(results['baseline_scratch'][str(k)]))) * 100
        for k in shot_sizes
    ])
    print(f"Average improvement vs Scratch: {avg_improvement_vs_scratch:+.1f}%")
    
    if avg_improvement_vs_scratch > 10:
        print("\n[SUCCESS] POC PASSED: MAML beats training from scratch")
        print(f"   Improvement: {avg_improvement_vs_scratch:.1f}%")
        print("\nWhy MAML Wins:")
        print("   - Meta-learning enables effective few-shot adaptation")
        print("   - Outperforms naive training on limited data")
        print("   - Shows promise for personalization")
    elif avg_improvement_vs_random > 10:
        print("\n[PARTIAL SUCCESS] MAML beats random baseline")
        print(f"   vs Random: {avg_improvement_vs_random:.1f}%")
        print(f"   vs Scratch: {avg_improvement_vs_scratch:.1f}%")
    else:
        print("\n[WEAK] POC INCONCLUSIVE: MAML does not clearly outperform baselines")
        print(f"   vs Random: {avg_improvement_vs_random:.1f}%")
        print(f"   vs Scratch: {avg_improvement_vs_scratch:.1f}%")

print("="*80 + "\n")

plt.show()

