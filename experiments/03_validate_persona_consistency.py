import sys
import os
sys.path.append('.')
from models.casino_persona_simulator import CasinoPersonaSimulator
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(texts):
    """Get embeddings using OpenAI API"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in response.data])

def test_persona_consistency(persona_type, test_prompt, num_trials=5):
    """Test if persona gives consistent responses"""
    
    print(f"\nTesting {persona_type} persona consistency...")
    
    responses = []
    for trial in range(num_trials):
        simulator = CasinoPersonaSimulator(persona_type, model="gpt-4o-mini")
        response = simulator.get_response(test_prompt)
        responses.append(response)
        print(f"  Trial {trial+1}: {response}")
    
    # Measure similarity using embeddings
    embeddings = get_embeddings(responses)
    
    # Compute pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            similarities.append(sim)
    
    avg_similarity = np.mean(similarities)
    print(f"  Average pairwise similarity: {avg_similarity:.3f}")
    
    return {
        'persona': persona_type,
        'responses': responses,
        'avg_similarity': float(avg_similarity),
        'embeddings': embeddings
    }

def test_inter_persona_diversity(test_prompt):
    """Test that different personas give different responses"""
    
    print(f"\nTesting inter-persona diversity...")
    
    personas = ['competitive_bargainer', 'empathetic_trader', 'strategic_negotiator', 
                'flexible_collaborator', 'assertive_claimer']
    persona_responses = {}
    
    for persona_type in personas:
        simulator = CasinoPersonaSimulator(persona_type, model="gpt-4o-mini")
        response = simulator.get_response(test_prompt)
        persona_responses[persona_type] = response
        print(f"  {persona_type}: {response}")
    
    # Measure diversity using embeddings
    responses_list = list(persona_responses.values())
    embeddings = get_embeddings(responses_list)
    
    # Compute average distance between personas
    distances = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = 1 - cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            distances.append(dist)
    
    avg_distance = np.mean(distances)
    print(f"  Average inter-persona distance: {avg_distance:.3f}")
    
    return {
        'persona_responses': persona_responses,
        'avg_distance': float(avg_distance),
        'embeddings': embeddings
    }

if __name__ == '__main__':
    test_prompt = "I really need firewood for my group. What supplies are you most interested in?"
    
    # Test consistency within each persona
    consistency_results = []
    for persona in ['competitive_bargainer', 'empathetic_trader', 'strategic_negotiator', 
                    'flexible_collaborator', 'assertive_claimer']:
        result = test_persona_consistency(persona, test_prompt, num_trials=5)
        consistency_results.append(result)
    
    # Test diversity between personas
    diversity_result = test_inter_persona_diversity(test_prompt)
    
    # Save results
    results = {
        'consistency': consistency_results,
        'diversity': diversity_result
    }
    
    with open('results/persona_validation.json', 'w') as f:
        # Remove embeddings for JSON serialization
        save_results = {
            'consistency': [
                {k: v for k, v in r.items() if k != 'embeddings'}
                for r in consistency_results
            ],
            'diversity': {
                k: v for k, v in diversity_result.items() if k != 'embeddings'
            }
        }
        json.dump(save_results, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Within-persona consistency (average): {np.mean([r['avg_similarity'] for r in consistency_results]):.3f}")
    print(f"Between-persona diversity: {diversity_result['avg_distance']:.3f}")
    print("\nResults saved to results/persona_validation.json")