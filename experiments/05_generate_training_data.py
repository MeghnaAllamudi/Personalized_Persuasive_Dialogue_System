import sys
sys.path.append('.')
from models.strategy_generator import StrategyPromptedGenerator
from models.persona_simulator import CasinoPersonaSimulator
import random
import json
import numpy as np
from textblob import TextBlob
from tqdm import tqdm

def compute_reward(user_response, turn_number):
    """
    Compute reward signal based on user response
    
    Positive indicators: decreased negativity, cooperation, agreement
    Negative indicators: increased aggression, resistance, escalation
    """
    sentiment = TextBlob(user_response).sentiment.polarity
    
    # Reward components
    sentiment_reward = sentiment  # Range: -1 to 1
    
    # Bonus for positive keywords
    positive_keywords = ['okay', 'understand', 'thanks', 'agree', 'yes', 'sure', 'appreciate']
    negative_keywords = ['no', 'never', 'unfair', 'wrong', 'angry', 'ridiculous', 'stupid']
    
    positive_count = sum(1 for word in positive_keywords if word in user_response.lower())
    negative_count = sum(1 for word in negative_keywords if word in user_response.lower())
    
    keyword_reward = (positive_count - negative_count) * 0.2
    
    # Early deescalation bonus
    turn_bonus = (5 - turn_number) * 0.1 if sentiment > 0 else 0
    
    total_reward = sentiment_reward + keyword_reward + turn_bonus
    
    # Clip to reasonable range
    return np.clip(total_reward, -2, 2)

def generate_episode(persona_type, episode_id, num_turns=5):
    """Generate a single training episode"""
    
    generator = StrategyPromptedGenerator(model="gpt-4o-mini")
    simulator = CasinoPersonaSimulator(persona_type, model="gpt-4o-mini")
    
    # Available strategies
    strategies_pool = ['empathy', 'validation', 'active_listening', 'problem_solving', 'authority']
    
    conversation = []
    cumulative_reward = 0
    
    for turn in range(num_turns):
        # Agent selects random strategy combination (1-2 strategies)
        num_strategies = random.randint(1, 2)
        selected_strategies = random.sample(strategies_pool, num_strategies)
        
        # Generate agent response
        agent_response = generator.generate_response(conversation, selected_strategies)
        
        # Get user response
        user_response = simulator.get_response(agent_response)
        
        # Compute reward
        reward = compute_reward(user_response, turn)
        cumulative_reward += reward
        
        # Record turn
        turn_data = {
            'turn': turn,
            'strategies': selected_strategies,
            'agent_message': agent_response,
            'user_message': user_response,
            'reward': float(reward),
            'cumulative_reward': float(cumulative_reward)
        }
        
        conversation.append(turn_data)
    
    return {
        'episode_id': episode_id,
        'persona_type': persona_type,
        'conversation': conversation,
        'total_reward': float(cumulative_reward),
        'final_sentiment': float(TextBlob(conversation[-1]['user_message']).sentiment.polarity)
    }

def generate_training_dataset(episodes_per_persona=20):
    """Generate full training dataset"""
    
    personas = ['competitive_bargainer', 'empathetic_trader', 'strategic_negotiator', 
                'flexible_collaborator', 'assertive_claimer']
    
    all_episodes = []
    episode_counter = 0
    
    print(f"Generating {episodes_per_persona} episodes per persona ({len(personas)} personas)...")
    print(f"Total episodes: {episodes_per_persona * len(personas)}")
    
    for persona in personas:
        print(f"\nGenerating episodes for {persona} persona...")
        
        for i in tqdm(range(episodes_per_persona)):
            episode = generate_episode(persona, episode_counter, num_turns=5)
            all_episodes.append(episode)
            episode_counter += 1
    
    # Save dataset
    with open('data/training_episodes.json', 'w') as f:
        json.dump(all_episodes, f, indent=2)
    
    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    for persona in personas:
        persona_episodes = [e for e in all_episodes if e['persona_type'] == persona]
        avg_reward = np.mean([e['total_reward'] for e in persona_episodes])
        avg_sentiment = np.mean([e['final_sentiment'] for e in persona_episodes])
        
        print(f"\n{persona.upper()}:")
        print(f"  Episodes: {len(persona_episodes)}")
        print(f"  Avg total reward: {avg_reward:.2f}")
        print(f"  Avg final sentiment: {avg_sentiment:.2f}")
    
    # Strategy effectiveness analysis
    strategy_rewards = {}
    for episode in all_episodes:
        for turn in episode['conversation']:
            for strategy in turn['strategies']:
                if strategy not in strategy_rewards:
                    strategy_rewards[strategy] = []
                strategy_rewards[strategy].append(turn['reward'])
    
    print(f"\nSTRATEGY EFFECTIVENESS (across all personas):")
    for strategy, rewards in sorted(strategy_rewards.items(), key=lambda x: np.mean(x[1]), reverse=True):
        print(f"  {strategy}: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    print(f"\nDataset saved to data/training_episodes.json")
    
    return all_episodes

if __name__ == '__main__':
    
    episodes = generate_training_dataset(episodes_per_persona=20)