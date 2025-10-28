import sys
sys.path.append('.')
from models.baselines import RandomStrategyAgent, PopulationBestAgent, PersonaSpecificAgent
from models.strategy_generator import StrategyPromptedGenerator
from models.casino_persona_simulator import CasinoPersonaSimulator
from textblob import TextBlob
import numpy as np
import json
from tqdm import tqdm

def compute_reward(user_response, turn_number):
    """Same reward function as training"""
    sentiment = TextBlob(user_response).sentiment.polarity
    
    positive_keywords = ['okay', 'understand', 'thanks', 'agree', 'yes', 'sure', 'appreciate']
    negative_keywords = ['no', 'never', 'unfair', 'wrong', 'angry', 'ridiculous', 'stupid']
    
    positive_count = sum(1 for word in positive_keywords if word in user_response.lower())
    negative_count = sum(1 for word in negative_keywords if word in user_response.lower())
    
    keyword_reward = (positive_count - negative_count) * 0.2
    turn_bonus = (5 - turn_number) * 0.1 if sentiment > 0 else 0
    
    total_reward = sentiment + keyword_reward + turn_bonus
    return np.clip(total_reward, -2, 2)

def evaluate_agent(agent, persona_type, num_episodes=10):
    """Evaluate agent on persona"""
    
    generator = StrategyPromptedGenerator(model="gpt-4o-mini")
    episode_rewards = []
    episode_sentiments = []
    
    for _ in range(num_episodes):
        simulator = CasinoPersonaSimulator(persona_type, model="gpt-4o-mini")
        conversation = []
        total_reward = 0
        
        for turn in range(5):
            # Agent selects strategies
            if isinstance(agent, PersonaSpecificAgent):
                strategies = agent.select_strategies({}, persona_type)
            else:
                strategies = agent.select_strategies({})
            
            # Generate response
            agent_msg = generator.generate_response(conversation, strategies)
            user_msg = simulator.get_response(agent_msg)
            
            reward = compute_reward(user_msg, turn)
            total_reward += reward
            
            conversation.append({'agent': agent_msg, 'user': user_msg})
        
        episode_rewards.append(total_reward)
        final_sentiment = TextBlob(conversation[-1]['user']).sentiment.polarity
        episode_sentiments.append(final_sentiment)
    
    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_sentiment': float(np.mean(episode_sentiments)),
        'std_sentiment': float(np.std(episode_sentiments))
    }

def run_evaluation():
    """Evaluate all baselines on all personas"""
    
    personas = ['competitive_bargainer', 'empathetic_trader', 'strategic_negotiator', 
                'flexible_collaborator', 'assertive_claimer']
    
    agents = {
        'random': RandomStrategyAgent(),
        'population_best': PopulationBestAgent(),
        'oracle': PersonaSpecificAgent()
    }
    
    results = {}
    
    print("Evaluating baseline agents...")
    print(f"Episodes per (agent, persona): 10")
    print(f"Total evaluations: {len(agents) * len(personas) * 10}\n")
    
    for agent_name, agent in agents.items():
        print(f"\nEvaluating {agent.name()}...")
        results[agent_name] = {}
        
        for persona in tqdm(personas):
            persona_results = evaluate_agent(agent, persona, num_episodes=10)
            results[agent_name][persona] = persona_results
            
            print(f"  {persona}: reward={persona_results['mean_reward']:.2f}±{persona_results['std_reward']:.2f}, "
                  f"sentiment={persona_results['mean_sentiment']:.2f}±{persona_results['std_sentiment']:.2f}")
    
    # Save results
    with open('results/baseline_performance.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE PERFORMANCE SUMMARY")
    print("="*80)
    
    for agent_name in agents:
        all_rewards = [results[agent_name][p]['mean_reward'] for p in personas]
        print(f"\n{agents[agent_name].name()}:")
        print(f"  Average reward across personas: {np.mean(all_rewards):.2f}")
        print(f"  Best persona: {max([(p, results[agent_name][p]['mean_reward']) for p in personas], key=lambda x: x[1])}")
        print(f"  Worst persona: {min([(p, results[agent_name][p]['mean_reward']) for p in personas], key=lambda x: x[1])}")
    
    return results

if __name__ == '__main__':
    results = run_evaluation()