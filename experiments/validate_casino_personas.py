import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI
sys.path.append('.')
from models.casino_persona_simulator import CasinoPersonaSimulator

# Load environment variables
load_dotenv()

# Set style
sns.set_style("whitegrid")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(texts, batch_size=100):
    """Get embeddings using OpenAI API"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Encoding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def load_casino_responses(num_dialogues=50):
    """Extract human responses from casino dataset"""
    with open('data/casino_processed.json') as f:
        dialogues = json.load(f)
    
    human_responses = []
    for dialogue in dialogues[:num_dialogues]:
        for turn in dialogue['turns']:
            text = turn['text']
            if len(text.split()) > 3:  # Filter out very short responses
                human_responses.append({
                    'text': text,
                    'source': 'human',
                    'dialogue_id': dialogue['dialogue_id']
                })
    
    return human_responses

def generate_casino_llm_responses(num_responses=100):
    """Generate LLM responses using casino-specific personas"""
    personas = ['competitive_bargainer', 'empathetic_trader', 'strategic_negotiator', 
                'flexible_collaborator', 'assertive_claimer']
    
    # Prompts that match casino negotiation context
    prompts = [
        "Hello! What supplies are you most interested in?",
        "I really need firewood to keep my family warm.",
        "What do you need most? Maybe we can work something out.",
        "I'm willing to give up all the water if I can get food and firewood.",
        "How about 2 firewood for 2 water and 1 food?",
        "I have young kids with me and need extra food packages.",
        "What are you least interested in?",
        "I definitely need at least 2 packages of food for my group.",
        "Can we trade? I'll give you what you need most.",
        "I need water because I forgot to bring enough."
    ]
    
    llm_responses = []
    for i in range(num_responses):
        persona = personas[i % len(personas)]
        prompt = prompts[i % len(prompts)]
        
        simulator = CasinoPersonaSimulator(persona, model="gpt-4o-mini")
        response = simulator.get_response(prompt)
        
        llm_responses.append({
            'text': response,
            'source': 'llm',
            'persona': persona
        })
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{num_responses} LLM responses...")
    
    return llm_responses

def compute_similarity_metrics(human_responses, llm_responses):
    """Compute various similarity metrics between human and LLM responses"""
    
    # Get embeddings
    human_texts = [r['text'] for r in human_responses]
    llm_texts = [r['text'] for r in llm_responses]
    
    print("Encoding human responses...")
    human_embeddings = get_embeddings(human_texts)
    
    print("Encoding LLM responses...")
    llm_embeddings = get_embeddings(llm_texts)
    
    # 1. Average similarity between human and LLM responses
    print("Computing cross-similarity...")
    cross_similarities = cosine_similarity(human_embeddings, llm_embeddings)
    avg_cross_similarity = np.mean(cross_similarities)
    
    # 2. Within-group similarities (human-to-human, llm-to-llm)
    print("Computing within-group similarities...")
    human_similarities = cosine_similarity(human_embeddings, human_embeddings)
    # Get upper triangle (exclude diagonal)
    human_sim_values = human_similarities[np.triu_indices_from(human_similarities, k=1)]
    avg_human_similarity = np.mean(human_sim_values)
    
    llm_similarities = cosine_similarity(llm_embeddings, llm_embeddings)
    llm_sim_values = llm_similarities[np.triu_indices_from(llm_similarities, k=1)]
    avg_llm_similarity = np.mean(llm_sim_values)
    
    # 3. Nearest neighbor analysis: for each LLM response, find closest human response
    print("Computing nearest neighbor similarities...")
    nearest_neighbor_sims = []
    nearest_neighbor_indices = []
    for llm_emb in llm_embeddings:
        sims = cosine_similarity([llm_emb], human_embeddings)[0]
        max_idx = np.argmax(sims)
        nearest_neighbor_sims.append(sims[max_idx])
        nearest_neighbor_indices.append(max_idx)
    avg_nn_similarity = np.mean(nearest_neighbor_sims)
    
    # 4. Per-persona nearest neighbor analysis
    print("Computing per-persona nearest neighbor similarities...")
    personas = ['competitive_bargainer', 'empathetic_trader', 'strategic_negotiator', 
                'flexible_collaborator', 'assertive_claimer']
    persona_nn_similarities = {}
    
    for persona in personas:
        persona_indices = [i for i, r in enumerate(llm_responses) if r['persona'] == persona]
        if persona_indices:
            persona_sims = [nearest_neighbor_sims[i] for i in persona_indices]
            persona_nn_similarities[persona] = {
                'avg': np.mean(persona_sims),
                'std': np.std(persona_sims),
                'min': np.min(persona_sims),
                'max': np.max(persona_sims),
                'all_sims': persona_sims
            }
    
    return {
        'avg_cross_similarity': avg_cross_similarity,
        'avg_human_similarity': avg_human_similarity,
        'avg_llm_similarity': avg_llm_similarity,
        'avg_nn_similarity': avg_nn_similarity,
        'nearest_neighbor_sims': nearest_neighbor_sims,
        'nearest_neighbor_indices': nearest_neighbor_indices,
        'persona_nn_similarities': persona_nn_similarities,
        'human_embeddings': human_embeddings,
        'llm_embeddings': llm_embeddings
    }

def create_visualization(human_responses, llm_responses, metrics):
    """Create multi-panel visualization comparing human and LLM responses"""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Similarity metrics comparison
    metric_names = ['Human-LLM\nCross', 'Human-Human\nWithin', 'LLM-LLM\nWithin', 'Nearest\nNeighbor']
    metric_values = [
        metrics['avg_cross_similarity'],
        metrics['avg_human_similarity'],
        metrics['avg_llm_similarity'],
        metrics['avg_nn_similarity']
    ]
    
    bars = axes[0].bar(metric_names, metric_values, 
                       color=['steelblue', 'lightcoral', 'lightgreen', 'orange'],
                       alpha=0.7)
    axes[0].set_ylabel('Average Cosine Similarity', fontsize=11)
    axes[0].set_title('(A) Human vs LLM Response Similarity', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Panel B: t-SNE visualization of human vs LLM embeddings
    print("Computing t-SNE...")
    all_embeddings = np.vstack([metrics['human_embeddings'], metrics['llm_embeddings']])
    
    # Sample if too many points
    max_points = 200
    if len(all_embeddings) > max_points:
        indices = np.random.choice(len(all_embeddings), max_points, replace=False)
        sampled_embeddings = all_embeddings[indices]
        sampled_labels = (['human'] * len(human_responses) + ['llm'] * len(llm_responses))
        sampled_labels = [sampled_labels[i] for i in indices]
    else:
        sampled_embeddings = all_embeddings
        sampled_labels = ['human'] * len(human_responses) + ['llm'] * len(llm_responses)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sampled_embeddings)-1))
    embeddings_2d = tsne.fit_transform(sampled_embeddings)
    
    # Plot human vs LLM
    human_mask = np.array(sampled_labels) == 'human'
    llm_mask = np.array(sampled_labels) == 'llm'
    
    axes[1].scatter(embeddings_2d[human_mask, 0], embeddings_2d[human_mask, 1],
                   label='Human (Casino)', color='coral', alpha=0.6, s=100, marker='o')
    axes[1].scatter(embeddings_2d[llm_mask, 0], embeddings_2d[llm_mask, 1],
                   label='LLM (Casino Persona)', color='steelblue', alpha=0.6, s=100, marker='^')
    
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=11)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=11)
    axes[1].set_title('(B) Human vs LLM Response Embeddings', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    
    # Panel C: Per-persona nearest neighbor similarity to human responses
    persona_nn = metrics['persona_nn_similarities']
    personas = sorted(persona_nn.keys())
    persona_avgs = [persona_nn[p]['avg'] for p in personas]
    persona_stds = [persona_nn[p]['std'] for p in personas]
    
    # Shorten names for display
    persona_labels = [p.replace('_', '\n') for p in personas]
    
    colors_map = {
        'assertive_claimer': '#e74c3c',
        'competitive_bargainer': '#f39c12',
        'empathetic_trader': '#2ecc71', 
        'flexible_collaborator': '#3498db',
        'strategic_negotiator': '#9b59b6'
    }
    bar_colors = [colors_map[p] for p in personas]
    
    bars = axes[2].bar(range(len(personas)), persona_avgs, yerr=persona_stds, 
                       color=bar_colors, alpha=0.7, capsize=5)
    axes[2].set_xticks(range(len(personas)))
    axes[2].set_xticklabels(persona_labels, fontsize=8)
    axes[2].set_ylabel('Average Nearest Neighbor Similarity', fontsize=11)
    axes[2].set_xlabel('Casino Negotiation Persona', fontsize=11)
    axes[2].set_title('(C) Per-Persona Similarity to Human Responses', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, 1)
    axes[2].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='High similarity')
    axes[2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Moderate similarity')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].legend(loc='upper right', fontsize=9)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/casino_persona_validation.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to results/casino_persona_validation.png")
    
    return fig

if __name__ == '__main__':
    print("="*80)
    print("VALIDATING CASINO-SPECIFIC PERSONAS AGAINST HUMAN CASINO DATASET")
    print("="*80)
    
    # Load human responses from casino dataset
    print("\nLoading human responses from casino dataset...")
    human_responses = load_casino_responses(num_dialogues=30)
    print(f"Loaded {len(human_responses)} human responses")
    
    # Generate LLM responses with casino-specific personas
    print("\nGenerating casino persona LLM responses...")
    llm_responses = generate_casino_llm_responses(num_responses=100)
    print(f"Generated {len(llm_responses)} LLM responses")
    
    # Compute similarity metrics
    print("\nComputing similarity metrics...")
    metrics = compute_similarity_metrics(human_responses, llm_responses)
    
    # Create visualization
    print("\nCreating visualization...")
    create_visualization(human_responses, llm_responses, metrics)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: CASINO-SPECIFIC PERSONAS")
    print("="*80)
    print(f"Human-LLM cross-similarity:     {metrics['avg_cross_similarity']:.3f}")
    print(f"Human-Human within-similarity:  {metrics['avg_human_similarity']:.3f}")
    print(f"LLM-LLM within-similarity:      {metrics['avg_llm_similarity']:.3f}")
    print(f"LLM nearest neighbor similarity: {metrics['avg_nn_similarity']:.3f}")
    
    print("\n" + "-"*80)
    print("PER-PERSONA NEAREST NEIGHBOR SIMILARITY TO HUMAN RESPONSES")
    print("-"*80)
    for persona, stats in sorted(metrics['persona_nn_similarities'].items()):
        print(f"{persona:25s}: {stats['avg']:.3f} (±{stats['std']:.3f}) | min={stats['min']:.3f}, max={stats['max']:.3f}")
    
    print("\nInterpretation:")
    print("- Casino-specific personas are designed to match negotiation context")
    print("- Higher similarities indicate more realistic negotiation behavior")
    print("- Compare these results with generic personas to see improvement")
    
    print("\nGuidelines:")
    if metrics['avg_nn_similarity'] > 0.7:
        print("✓ EXCELLENT: Casino personas are very similar to human negotiators")
    elif metrics['avg_nn_similarity'] > 0.5:
        print("✓ GOOD: Casino personas show solid similarity to human negotiators")
    elif metrics['avg_nn_similarity'] > 0.4:
        print("✓ ACCEPTABLE: Casino personas show moderate similarity to humans")
    else:
        print("⚠ NEEDS WORK: Casino personas need more refinement")
    
    # Show example best matches for each persona
    print("\n" + "-"*80)
    print("EXAMPLE BEST MATCHES (LLM → Nearest Human Response)")
    print("-"*80)
    personas = ['competitive_bargainer', 'empathetic_trader', 'strategic_negotiator', 
                'flexible_collaborator', 'assertive_claimer']
    for persona in personas:
        # Find the best match for this persona
        persona_indices = [i for i, r in enumerate(llm_responses) if r['persona'] == persona]
        if persona_indices:
            best_idx = max(persona_indices, key=lambda i: metrics['nearest_neighbor_sims'][i])
            best_sim = metrics['nearest_neighbor_sims'][best_idx]
            human_idx = metrics['nearest_neighbor_indices'][best_idx]
            
            print(f"\n{persona.upper().replace('_', ' ')} (similarity: {best_sim:.3f})")
            print(f"  LLM: {llm_responses[best_idx]['text'][:120]}...")
            print(f"  Human: {human_responses[human_idx]['text'][:120]}...")
    
    # Save metrics
    save_metrics = {
        'avg_cross_similarity': metrics['avg_cross_similarity'],
        'avg_human_similarity': metrics['avg_human_similarity'],
        'avg_llm_similarity': metrics['avg_llm_similarity'],
        'avg_nn_similarity': metrics['avg_nn_similarity'],
        'persona_nn_similarities': {
            persona: {
                'avg': float(stats['avg']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max'])
            }
            for persona, stats in metrics['persona_nn_similarities'].items()
        }
    }
    with open('results/casino_persona_metrics.json', 'w') as f:
        json.dump(save_metrics, f, indent=2)
    
    print("\nMetrics saved to results/casino_persona_metrics.json")

