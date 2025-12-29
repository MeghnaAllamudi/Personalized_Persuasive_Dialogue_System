
import random
import numpy as np
import json
from collections import Counter

class RandomStrategyAgent:
    """Baseline: Select strategies randomly"""
    
    def __init__(self, strategies_pool=['empathy', 'validation', 'active_listening', 'problem_solving', 'authority']):
        self.strategies_pool = strategies_pool
    
    def select_strategies(self, conversation_state):
        """Select random strategy combination"""
        num_strategies = random.randint(1, 2)
        return random.sample(self.strategies_pool, num_strategies)
    
    def name(self):
        return "Random Strategy Selection"


class PopulationBestAgent:
    """Baseline: Use the single best strategy across all personas"""
    
    def __init__(self, training_data_path='data/training_episodes.json'):
        # Compute which strategy has highest average reward
        with open(training_data_path) as f:
            episodes = json.load(f)
        
        strategy_rewards = {}
        for episode in episodes:
            for turn in episode['conversation']:
                for strategy in turn['strategies']:
                    if strategy not in strategy_rewards:
                        strategy_rewards[strategy] = []
                    strategy_rewards[strategy].append(turn['reward'])
        
        # Find best strategy
        avg_rewards = {s: np.mean(rewards) for s, rewards in strategy_rewards.items()}
        self.best_strategy = max(avg_rewards, key=avg_rewards.get)
        
        print(f"Population-best strategy: {self.best_strategy} (avg reward: {avg_rewards[self.best_strategy]:.2f})")
    
    def select_strategies(self, conversation_state):
        """Always use the population-best strategy"""
        return [self.best_strategy]
    
    def name(self):
        return f"Population-Best ({self.best_strategy})"


class PersonaSpecificAgent:
    """Baseline: Pre-compute best strategy for each persona (oracle)"""
    
    def __init__(self, training_data_path='data/training_episodes.json'):
        with open(training_data_path) as f:
            episodes = json.load(f)
        
        # Compute best strategy for each persona
        self.persona_strategies = {}
        
        personas = set(e['persona_type'] for e in episodes)
        for persona in personas:
            persona_episodes = [e for e in episodes if e['persona_type'] == persona]
            
            strategy_rewards = {}
            for episode in persona_episodes:
                for turn in episode['conversation']:
                    for strategy in turn['strategies']:
                        if strategy not in strategy_rewards:
                            strategy_rewards[strategy] = []
                        strategy_rewards[strategy].append(turn['reward'])
            
            avg_rewards = {s: np.mean(rewards) for s, rewards in strategy_rewards.items()}
            best_strategy = max(avg_rewards, key=avg_rewards.get)
            self.persona_strategies[persona] = best_strategy
            
            print(f"Best strategy for {persona}: {best_strategy} (avg reward: {avg_rewards[best_strategy]:.2f})")
    
    def select_strategies(self, conversation_state, persona_type):
        """Use best strategy for this specific persona (requires knowing persona)"""
        return [self.persona_strategies.get(persona_type, 'empathy')]
    
    def name(self):
        return "Persona-Specific Oracle"


class MostCommonStrategyAgent:
    """Baseline: Use the most frequently used strategy"""
    
    def __init__(self, training_data_path='data/training_episodes.json'):
        with open(training_data_path) as f:
            episodes = json.load(f)
        
        all_strategies = []
        for episode in episodes:
            for turn in episode['conversation']:
                all_strategies.extend(turn['strategies'])
        
        strategy_counts = Counter(all_strategies)
        self.most_common = strategy_counts.most_common(1)[0][0]
        
        print(f"Most common strategy: {self.most_common} (appeared {strategy_counts[self.most_common]} times)")
    
    def select_strategies(self, conversation_state):
        """Always use most common strategy"""
        return [self.most_common]
    
    def name(self):
        return f"Most Common ({self.most_common})"


class VanillaLLMAgent:
    """Baseline: No strategy guidance - just prompt LLM to persuade (similar to PersuaBot)"""
    
    def __init__(self):
        print("Vanilla LLM: No strategy selection (direct persuasion prompting)")
    
    def select_strategies(self, conversation_state):
        """Return empty list to signal no strategy guidance"""
        return []
    
    def name(self):
        return "Vanilla LLM (No Strategy)"


# Test baselines
if __name__ == '__main__':
    print("Initializing baseline agents...\n")
    
    random_agent = RandomStrategyAgent()
    population_agent = PopulationBestAgent()
    oracle_agent = PersonaSpecificAgent()
    common_agent = MostCommonStrategyAgent()
    vanilla_agent = VanillaLLMAgent()
    
    print("\n" + "="*80)
    print("Testing strategy selection:")
    print("="*80)
    
    test_state = {}  # Dummy state
    
    print(f"\n{random_agent.name()}: {random_agent.select_strategies(test_state)}")
    print(f"{population_agent.name()}: {population_agent.select_strategies(test_state)}")
    print(f"{oracle_agent.name()} [competitive_bargainer]: {oracle_agent.select_strategies(test_state, 'competitive_bargainer')}")
    print(f"{common_agent.name()}: {common_agent.select_strategies(test_state)}")
    print(f"{vanilla_agent.name()}: {vanilla_agent.select_strategies(test_state)}")