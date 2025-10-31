"""
Generate sample conversations to showcase differences between methods
Saves actual LLM-generated dialogues for qualitative analysis
"""
import sys
sys.path.append('.')
import torch
import numpy as np
import json
from models.strategy_selector import StrategySelector
from models.maml_trainer import MAMLTrainer
from models.strategy_generator import StrategyPromptedGenerator
from models.casino_persona_simulator import CasinoPersonaSimulator
from models.baselines import RandomStrategyAgent, PopulationBestAgent, PersonaSpecificAgent
from models.state_encoder import ConversationStateEncoder
from textblob import TextBlob

def compute_reward(user_response, turn_number):
    """Compute reward for a user response"""
    sentiment = TextBlob(user_response).sentiment.polarity
    positive_keywords = ['okay', 'understand', 'thanks', 'agree', 'yes', 'sure', 'appreciate']
    negative_keywords = ['no', 'never', 'unfair', 'wrong', 'angry', 'ridiculous', 'stupid']
    
    positive_count = sum(1 for word in positive_keywords if word in user_response.lower())
    negative_count = sum(1 for word in negative_keywords if word in user_response.lower())
    keyword_reward = (positive_count - negative_count) * 0.2
    turn_bonus = (5 - turn_number) * 0.1 if sentiment > 0 else 0
    
    return np.clip(sentiment + keyword_reward + turn_bonus, -2, 2)

def generate_sample_conversation(method_name, agent, persona_type, num_shots=3, 
                                 adapted_params=None, trainer=None, encoder=None):
    """Generate one sample conversation for a given method"""
    
    strategies = ['empathy', 'validation', 'active_listening', 'problem_solving', 'authority']
    strategy_to_idx = {s: i for i, s in enumerate(strategies)}
    
    generator = StrategyPromptedGenerator(model="gpt-4o-mini")
    simulator = CasinoPersonaSimulator(persona_type, model="gpt-4o-mini")
    
    conversation = []
    total_reward = 0
    
    print(f"\nGenerating {method_name} conversation for {persona_type}...")
    
    for turn in range(5):
        # Select strategies based on method
        if method_name == "MAML":
            state = encoder.encode_conversation(conversation)
            probs = trainer.predict_with_params(adapted_params, state)
            indices = np.where(probs > 0.5)[0]
            if len(indices) == 0:
                indices = [np.argmax(probs)]
            selected_strategies = [strategies[i] for i in indices]
        elif method_name == "Oracle":
            selected_strategies = agent.select_strategies({}, persona_type)
        else:  # Random or Population-Best
            selected_strategies = agent.select_strategies({})
        
        # Generate response
        agent_msg = generator.generate_response(conversation, selected_strategies)
        user_msg = simulator.get_response(agent_msg)
        
        # Compute reward
        reward = compute_reward(user_msg, turn)
        total_reward += reward
        
        # Get sentiment
        sentiment = TextBlob(user_msg).sentiment.polarity
        
        conversation.append({
            'turn': turn + 1,
            'strategies': selected_strategies,
            'agent_message': agent_msg,
            'user_message': user_msg,
            'reward': float(reward),
            'sentiment': float(sentiment),
            'cumulative_reward': float(total_reward)
        })
        
        print(f"  Turn {turn + 1}: {selected_strategies} -> reward={reward:.2f}")
    
    return {
        'method': method_name,
        'persona_type': persona_type,
        'num_adaptation_shots': num_shots if method_name == "MAML" else "N/A",
        'conversation': conversation,
        'total_reward': float(total_reward),
        'final_sentiment': conversation[-1]['sentiment']
    }

def generate_maml_conversation_with_adaptation(persona_type, num_shots=3):
    """Generate MAML conversation with adaptation phase shown"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = StrategySelector(state_dim=384, num_strategies=5)
    model.load_state_dict(torch.load('results/maml_model.pt', map_location=device))
    model.to(device)
    model.eval()
    
    trainer = MAMLTrainer(model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=5, device=device)
    encoder = ConversationStateEncoder(device=device)
    
    strategies = ['empathy', 'validation', 'active_listening', 'problem_solving', 'authority']
    strategy_to_idx = {s: i for i, s in enumerate(strategies)}
    
    generator = StrategyPromptedGenerator(model="gpt-4o-mini")
    
    # Phase 1: Adaptation (support set)
    print(f"\n{'='*80}")
    print(f"MAML ADAPTATION PHASE ({num_shots} shots)")
    print(f"{'='*80}")
    
    simulator = CasinoPersonaSimulator(persona_type, model="gpt-4o-mini")
    support_data = []
    adaptation_conversation = []
    
    for turn_idx in range(num_shots):
        # Use random strategies for support set
        selected_strats = [strategies[np.random.randint(0, 5)]]
        agent_msg = generator.generate_response(adaptation_conversation, selected_strats)
        user_msg = simulator.get_response(agent_msg)
        reward = compute_reward(user_msg, turn_idx)
        
        state = encoder.encode_conversation(adaptation_conversation)
        action = np.zeros(5)
        for s in selected_strats:
            action[strategy_to_idx[s]] = 1
        
        support_data.append({
            'state': state,
            'action': action,
            'reward': reward
        })
        
        adaptation_conversation.append({
            'turn': turn_idx + 1,
            'strategies': selected_strats,
            'agent_message': agent_msg,
            'user_message': user_msg,
            'reward': float(reward),
            'sentiment': float(TextBlob(user_msg).sentiment.polarity)
        })
        
        print(f"  Adaptation Turn {turn_idx + 1}: {selected_strats} -> reward={reward:.2f}")
    
    # Adapt the model
    adapted_params = trainer._inner_loop(support_data, create_graph=False)
    print("\nModel adapted! Now generating test conversation...")
    
    # Phase 2: Test with adapted model
    test_conversation = generate_sample_conversation(
        method_name="MAML",
        agent=None,
        persona_type=persona_type,
        num_shots=num_shots,
        adapted_params=adapted_params,
        trainer=trainer,
        encoder=encoder
    )
    
    # Include adaptation phase in output
    test_conversation['adaptation_phase'] = adaptation_conversation
    
    return test_conversation

def main():
    """Generate sample conversations for all methods"""
    
    # Pick one persona for demonstration
    persona_type = 'empathetic_trader'
    num_shots = 3
    
    print(f"\n{'='*80}")
    print(f"GENERATING SAMPLE CONVERSATIONS")
    print(f"Persona: {persona_type}")
    print(f"MAML adaptation shots: {num_shots}")
    print(f"{'='*80}")
    
    samples = {}
    
    # 1. Generate MAML conversation (with adaptation)
    samples['maml'] = generate_maml_conversation_with_adaptation(persona_type, num_shots)
    
    # 2. Generate Random baseline conversation
    random_agent = RandomStrategyAgent()
    samples['random'] = generate_sample_conversation(
        method_name="Random",
        agent=random_agent,
        persona_type=persona_type
    )
    
    # 3. Generate Population-Best baseline conversation
    population_agent = PopulationBestAgent()
    samples['population_best'] = generate_sample_conversation(
        method_name="Population-Best",
        agent=population_agent,
        persona_type=persona_type
    )
    
    # 4. Generate Oracle baseline conversation
    oracle_agent = PersonaSpecificAgent()
    samples['oracle'] = generate_sample_conversation(
        method_name="Oracle",
        agent=oracle_agent,
        persona_type=persona_type
    )
    
    # Save to file
    output_file = 'results/sample_conversations.json'
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SAMPLE CONVERSATIONS SAVED")
    print(f"{'='*80}")
    print(f"File: {output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Total Reward':<15} {'Final Sentiment':<15}")
    print("-" * 50)
    for method_key, data in samples.items():
        print(f"{data['method']:<20} {data['total_reward']:<15.2f} {data['final_sentiment']:<15.3f}")
    
    # Create a readable text version too
    text_output = f"""
{'='*80}
SAMPLE CONVERSATION COMPARISON
Persona Type: {persona_type}
{'='*80}

"""
    
    for method_key, data in samples.items():
        text_output += f"\n{'='*80}\n"
        text_output += f"METHOD: {data['method']}\n"
        if data['method'] == 'MAML':
            text_output += f"Adaptation shots: {data['num_adaptation_shots']}\n"
            text_output += f"{'='*80}\n\n"
            text_output += f"--- ADAPTATION PHASE ({num_shots} shots) ---\n\n"
            for turn in data['adaptation_phase']:
                text_output += f"Turn {turn['turn']} [Strategies: {', '.join(turn['strategies'])}]\n"
                text_output += f"Agent: {turn['agent_message']}\n"
                text_output += f"User:  {turn['user_message']}\n"
                text_output += f"Reward: {turn['reward']:.2f}, Sentiment: {turn['sentiment']:.2f}\n\n"
            text_output += f"\n--- TEST PHASE (adapted model) ---\n\n"
        else:
            text_output += f"{'='*80}\n\n"
        
        for turn in data['conversation']:
            text_output += f"Turn {turn['turn']} [Strategies: {', '.join(turn['strategies'])}]\n"
            text_output += f"Agent: {turn['agent_message']}\n"
            text_output += f"User:  {turn['user_message']}\n"
            text_output += f"Reward: {turn['reward']:.2f}, Sentiment: {turn['sentiment']:.2f}, "
            text_output += f"Cumulative: {turn['cumulative_reward']:.2f}\n\n"
        
        text_output += f"\nFINAL RESULTS:\n"
        text_output += f"  Total Reward: {data['total_reward']:.2f}\n"
        text_output += f"  Final Sentiment: {data['final_sentiment']:.3f}\n"
    
    # Save text version
    with open('results/sample_conversations.txt', 'w', encoding='utf-8') as f:
        f.write(text_output)
    
    print(f"\nReadable version saved to: results/sample_conversations.txt")
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()

