import sys
sys.path.append('.')
from models.state_encoder import ConversationStateEncoder
import json
import pickle
import numpy as np

def prepare_maml_dataset():
    """
    Convert episode data into (state, action, reward) format for MAML
    
    Returns:
        List of tasks, where each task is a list of (state, action, reward) tuples
    """
    
    # Load episodes
    with open('data/training_episodes.json') as f:
        episodes = json.load(f)
    
    print(f"Loaded {len(episodes)} episodes")
    
    # Initialize encoder
    encoder = ConversationStateEncoder()
    
    # Strategy to index mapping
    strategies = ['empathy', 'validation', 'active_listening', 'problem_solving', 'authority']
    strategy_to_idx = {s: i for i, s in enumerate(strategies)}
    
    print(f"Strategy vocabulary: {strategies}")
    
    # Process each episode as a task
    tasks = []
    
    for episode in episodes:
        task_data = []  # One task = one persona's episode
        
        conversation_history = []
        
        for turn in episode['conversation']:
            # Encode current state
            state = encoder.encode_conversation(conversation_history)
            
            # Encode action (strategy combination as multi-hot vector)
            action = np.zeros(len(strategies))
            for strategy in turn['strategies']:
                if strategy in strategy_to_idx:
                    action[strategy_to_idx[strategy]] = 1
            
            # Get reward
            reward = turn['reward']
            
            task_data.append({
                'state': state,
                'action': action,
                'reward': reward,
                'turn': turn['turn'],
                'strategies': turn['strategies']
            })
            
            # Update history for next turn
            conversation_history.append({
                'agent': turn['agent_message'],
                'user': turn['user_message']
            })
        
        tasks.append({
            'persona': episode['persona_type'],
            'episode_id': episode['episode_id'],
            'data': task_data
        })
    
    print(f"\nPrepared {len(tasks)} tasks")
    print(f"State dimension: {tasks[0]['data'][0]['state'].shape}")
    print(f"Action dimension: {tasks[0]['data'][0]['action'].shape}")
    
    # Group tasks by persona
    persona_tasks = {}
    for task in tasks:
        persona = task['persona']
        if persona not in persona_tasks:
            persona_tasks[persona] = []
        persona_tasks[persona].append(task)
    
    print("\nTasks per persona:")
    for persona, task_list in persona_tasks.items():
        print(f"  {persona}: {len(task_list)} tasks")
    
    # Save processed data
    with open('data/maml_tasks.pkl', 'wb') as f:
        pickle.dump(tasks, f)
    
    print("\nSaved to data/maml_tasks.pkl")
    
    return tasks

if __name__ == '__main__':
    tasks = prepare_maml_dataset()
    
    # Show example task
    print("\n" + "="*80)
    print("EXAMPLE TASK")
    print("="*80)
    example = tasks[0]
    print(f"Persona: {example['persona']}")
    print(f"Episode ID: {example['episode_id']}")
    print(f"Number of turns: {len(example['data'])}")
    print("\nFirst turn:")
    print(f"  State shape: {example['data'][0]['state'].shape}")
    print(f"  Action (multi-hot): {example['data'][0]['action']}")
    print(f"  Strategies: {example['data'][0]['strategies']}")
    print(f"  Reward: {example['data'][0]['reward']:.2f}")

