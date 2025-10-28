import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import sys
sys.path.append('.')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 5)

# Load validation results
with open('results/persona_validation.json') as f:
    results = json.load(f)

# Create 3-panel figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Within-persona consistency
personas = [r['persona'] for r in results['consistency']]
consistency_scores = [r['avg_similarity'] for r in results['consistency']]

axes[0].bar(personas, consistency_scores, color='steelblue', alpha=0.7)
axes[0].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High consistency threshold')
axes[0].set_ylabel('Average Cosine Similarity', fontsize=11)
axes[0].set_xlabel('Persona Type', fontsize=11)
axes[0].set_title('(A) Within-Persona Response Consistency', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 1)
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# Panel B: t-SNE visualization of persona embeddings
# Re-generate embeddings for visualization  
from models.casino_persona_simulator import CasinoPersonaSimulator
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

test_prompt = "I really need firewood for my group. What supplies are you most interested in?"

all_responses = []
all_labels = []
for persona in personas:
    for trial in range(5):
        simulator = CasinoPersonaSimulator(persona, model="gpt-4o-mini")
        response = simulator.get_response(test_prompt)
        all_responses.append(response)
        all_labels.append(persona)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=all_responses
)
embeddings = np.array([item.embedding for item in response.data])
tsne = TSNE(n_components=2, random_state=42, perplexity=15)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot with different colors per persona
colors = plt.cm.Set2(np.linspace(0, 1, len(personas)))
for idx, persona in enumerate(personas):
    mask = np.array(all_labels) == persona
    axes[1].scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        label=persona,
        color=colors[idx],
        alpha=0.6,
        s=100
    )

axes[1].set_xlabel('t-SNE Dimension 1', fontsize=11)
axes[1].set_ylabel('t-SNE Dimension 2', fontsize=11)
axes[1].set_title('(B) Persona Embeddings Show Distinct Clusters', fontsize=12, fontweight='bold')
axes[1].legend(loc='best', fontsize=9)

# Panel C: Sentiment trajectories (simulated multi-turn)
from textblob import TextBlob

def get_sentiment(text):
    """Simple sentiment analysis"""
    return TextBlob(text).sentiment.polarity

# Simulate 5-turn conversations for each persona
from models.strategy_generator import StrategyPromptedGenerator

generator = StrategyPromptedGenerator(model="gpt-4o-mini")
sentiment_trajectories = {}

strategies_sequence = [
    ['empathy'],
    ['empathy', 'validation'],
    ['active_listening'],
    ['problem_solving'],
    ['validation']
]

for persona in personas:
    simulator = CasinoPersonaSimulator(persona, model="gpt-4o-mini")
    sentiments = []
    conversation = []
    
    for strategies in strategies_sequence:
        agent_msg = generator.generate_response(conversation, strategies)
        user_msg = simulator.get_response(agent_msg)
        
        sentiment = get_sentiment(user_msg)
        sentiments.append(sentiment)
        
        conversation.append({'agent': agent_msg, 'user': user_msg})
    
    sentiment_trajectories[persona] = sentiments

# Plot sentiment over turns
for idx, (persona, sentiments) in enumerate(sentiment_trajectories.items()):
    axes[2].plot(
        range(1, len(sentiments) + 1),
        sentiments,
        marker='o',
        label=persona,
        color=colors[idx],
        linewidth=2,
        markersize=8
    )

axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
axes[2].set_xlabel('Conversation Turn', fontsize=11)
axes[2].set_ylabel('Sentiment Polarity', fontsize=11)
axes[2].set_title('(C) Different Personas Show Distinct Deescalation Dynamics', fontsize=12, fontweight='bold')
axes[2].legend(loc='best', fontsize=9)
axes[2].set_ylim(-0.5, 0.5)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/persona_validation.png', dpi=300, bbox_inches='tight')
print("Figure saved to results/persona_validation.png")
plt.show()
