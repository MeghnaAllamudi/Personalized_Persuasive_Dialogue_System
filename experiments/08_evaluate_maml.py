import sys
sys.path.append('.')
import torch
import numpy as np
import pickle
import json
from models.strategy_selector import StrategySelector
from models.maml_trainer import MAMLTrainer
from models.strategy_generator import StrategyPromptedGenerator
from models.persona_simulator import PersonaSimulator
from textblob import TextBlob
from tqdm import tqdm

def compute_reward(user_response, turn_number):
    """Same reward function"""
    sentiment = TextBlob(user_response).sentiment.polarity
    positive_keywords = ['okay', 'understand', 'thanks', 'agree', 'yes', 'sure', 'appreciate']
    negative_keywords = ['no', 'never', 'unfair', 'wrong', 'angry', 'ridiculous', 'stupid']
    
    positive_count = sum(1 for word in positive_keywords if word in user_response.lower())
    negative_count = sum(1 for word in negative_keywords if word in user_response.lower())
    keyword_reward = (positive_count - negative_count) * 0.2
    turn_bonus = (5 - turn_number) * 0.1 if sentiment > 0 else 0
    
    return np.clip(sentiment + keyword_reward + turn_bonus, -2, 2)

def evaluate_few_shot_adaptation(model_path='results/maml_model.pt', num_shots=[1, 3, 5]):
    """
    Test MAML few-shot adaptation vs baselines
    
    For each test persona:
        - Adapt MAML using K examples
        - Compare against: random, population-best, training from scratch
    """
    
    # Load model
    model = StrategySelector(state_dim=384, num_strategies=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    trainer = MAMLTrainer(model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=5)
    
    # Test personas (held-out from training or new instances)
    test_personas = ['aggressive', 'cooperative', 'anxious', 'stubborn', 'diplomatic']
    
    strategies = ['empathy', 'validation', 'active_listening', 'problem_solving', 'authority']
    strategy_to_idx = {s: i for i, s in enumerate(strategies)}
    
    generator = StrategyPromptedGenerator(model="gpt-4o-mini")
    
    from models.state_encoder import ConversationStateEncoder
    encoder = ConversationStateEncoder()
    
    results = {
        'maml': {k: [] for k in num_shots},
        'baseline_random': {k: [] for k in num_shots},
        'baseline_scratch': {k: [] for k in num_shots}
    }
    
    print("Evaluating few-shot adaptation...\n")
    
    for persona_type in test_personas:
        print(f"\nTesting on {persona_type} persona...")
        
        for num_shot in num_shots:
            print(f"  K={num_shot} shots...")
            
            # Generate test episodes
            num_test_episodes = 5
            
            for episode_idx in range(num_test_episodes):
                # Generate adaptation data (support set)
                simulator = PersonaSimulator(persona_type, model="gpt-4o-mini")
                support_data = []
                conversation = []
                
                for turn_idx in range(num_shot):
                    # Use random strategy for support data
                    selected_strats = [strategies[np.random.randint(0, 5)]]
                    agent_msg = generator.generate_response(conversation, selected_strats)
                    user_msg = simulator.get_response(agent_msg, selected_strats)
                    reward = compute_reward(user_msg, turn_idx)
                    
                    state = encoder.encode_conversation(conversation)
                    action = np.zeros(5)
                    for s in selected_strats:
                        action[strategy_to_idx[s]] = 1
                    
                    support_data.append({
                        'state': state,
                        'action': action,
                        'reward': reward
                    })
                    
                    conversation.append({'agent': agent_msg, 'user': user_msg})
                
                # Now test adapted models
                # 1. MAML: Adapt using support data
                adapted_params = trainer._inner_loop(support_data, create_graph=False)
                
                # 2. Baseline: Random selection (no training needed)
                
                # 3. Baseline: Train new model from scratch on support data
                scratch_model = StrategySelector(state_dim=384, num_strategies=5)
                scratch_optimizer = torch.optim.Adam(scratch_model.parameters(), lr=0.01)
                
                # Train scratch model
                for _ in range(20):  # Train for 20 steps
                    total_loss = 0.0
                    for turn in support_data:
                        state_t = torch.tensor(turn['state'], dtype=torch.float32).unsqueeze(0)
                        action_t = torch.tensor(turn['action'], dtype=torch.float32).unsqueeze(0)
                        
                        logits = scratch_model(state_t)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, action_t, reduction='mean'
                        )
                        total_loss += loss
                    
                    total_loss = total_loss / len(support_data)
                    scratch_optimizer.zero_grad()
                    total_loss.backward()
                    scratch_optimizer.step()
                
                # Test on new episode
                test_simulator = PersonaSimulator(persona_type, model="gpt-4o-mini")
                test_conversation = []
                
                maml_reward = 0
                random_reward = 0
                scratch_reward = 0
                
                for test_turn in range(5):
                    state = encoder.encode_conversation(test_conversation)
                    
                    # MAML selection (use adapted parameters)
                    maml_probs = trainer.predict_with_params(adapted_params, state)
                    maml_indices = np.where(maml_probs > 0.5)[0]
                    if len(maml_indices) == 0:
                        maml_indices = [np.argmax(maml_probs)]
                    maml_strats = [strategies[i] for i in maml_indices]
                    
                    # Random selection
                    random_strats = [strategies[np.random.randint(0, 5)]]
                    
                    # Scratch model selection
                    scratch_indices, _ = scratch_model.select_strategies(state, threshold=0.5)
                    scratch_strats = [strategies[i] for i in scratch_indices]
                    
                    # Test each
                    # MAML
                    agent_msg = generator.generate_response(test_conversation, maml_strats)
                    user_msg = test_simulator.get_response(agent_msg, maml_strats)
                    maml_reward += compute_reward(user_msg, test_turn)
                    test_conversation.append({'agent': agent_msg, 'user': user_msg})
                    
                    # Random (reset simulator)
                    test_simulator_random = PersonaSimulator(persona_type, model="gpt-4o-mini")
                    test_conversation_random = list(test_conversation)
                    agent_msg_rand = generator.generate_response(test_conversation_random[:-1], random_strats)
                    user_msg_rand = test_simulator_random.get_response(agent_msg_rand, random_strats)
                    random_reward += compute_reward(user_msg_rand, test_turn)
                    
                    # Scratch (reset simulator)
                    test_simulator_scratch = PersonaSimulator(persona_type, model="gpt-4o-mini")
                    test_conversation_scratch = list(test_conversation)
                    agent_msg_scratch = generator.generate_response(test_conversation_scratch[:-1], scratch_strats)
                    user_msg_scratch = test_simulator_scratch.get_response(agent_msg_scratch, scratch_strats)
                    scratch_reward += compute_reward(user_msg_scratch, test_turn)
                
                # Record results
                results['maml'][num_shot].append(maml_reward)
                results['baseline_random'][num_shot].append(random_reward)
                results['baseline_scratch'][num_shot].append(scratch_reward)
    
    # Save results
    with open('results/few_shot_adaptation_results.json', 'w') as f:
        # Convert to regular floats for JSON
        save_results = {}
        for method in results:
            save_results[method] = {}
            for k in num_shots:
                save_results[method][k] = [float(x) for x in results[method][k]]
        json.dump(save_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("FEW-SHOT ADAPTATION RESULTS")
    print("="*80)
    
    for k in num_shots:
        print(f"\nK={k} shots:")
        for method in results:
            rewards = results[method][k]
            print(f"  {method:20s}: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        
        # Compute improvement
        maml_mean = np.mean(results['maml'][k])
        random_mean = np.mean(results['baseline_random'][k])
        scratch_mean = np.mean(results['baseline_scratch'][k])
        
        improvement_vs_random = (maml_mean - random_mean) / abs(random_mean) * 100 if random_mean != 0 else 0
        improvement_vs_scratch = (maml_mean - scratch_mean) / abs(scratch_mean) * 100 if scratch_mean != 0 else 0
        
        print(f"  MAML improvement vs random: {improvement_vs_random:.1f}%")
        print(f"  MAML improvement vs scratch: {improvement_vs_scratch:.1f}%")
    
    return results

if __name__ == '__main__':
    results = evaluate_few_shot_adaptation(
        model_path='results/maml_model.pt',
        num_shots=[1, 3, 5]
    )
