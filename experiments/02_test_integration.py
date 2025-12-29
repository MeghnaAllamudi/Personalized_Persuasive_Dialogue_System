import sys
sys.path.append('.')
from models.strategy_generator import StrategyPromptedGenerator
from models.persona_simulator import CasinoPersonaSimulator

def test_conversation(persona_type, strategy_sequence):
    """Test a full conversation with strategy-persona interaction"""
    
    print(f"\n{'='*80}")
    print(f"CONVERSATION: {persona_type.upper()} with strategies {strategy_sequence}")
    print('='*80)
    
    generator = StrategyPromptedGenerator(model="gpt-4o-mini")
    simulator = CasinoPersonaSimulator(persona_type, model="gpt-4o-mini")
    
    conversation_history = []
    
    # Initial greeting for casino negotiation
    initial_user_msg = "Hello! Let's talk about the camping supplies."
    print(f"\nUser (initial): {initial_user_msg}")
    
    # Add initial user message to history
    conversation_history.append({
        'user': initial_user_msg,
        'agent': ''
    })
    
    # Simulate 3 turns
    for turn_idx, strategies in enumerate(strategy_sequence):
        print(f"\n--- Turn {turn_idx + 1} ---")
        print(f"Strategies: {strategies}")
        
        # Agent responds
        agent_msg = generator.generate_response(conversation_history, strategies)
        print(f"Agent: {agent_msg}")
        
        # Persona responds
        user_msg = simulator.get_response(agent_msg)
        print(f"User: {user_msg}")
        
        # Update history (update last turn's agent response and add new user message)
        conversation_history[-1]['agent'] = agent_msg
        conversation_history.append({
            'user': user_msg,
            'agent': ''
        })
    
    return conversation_history

if __name__ == '__main__':
    # Test 1: Competitive bargainer with empathy -> validation
    test_conversation(
        'competitive_bargainer',
        [['empathy'], ['empathy', 'validation'], ['problem_solving']]
    )
    
    # Test 2: Empathetic trader with active listening
    test_conversation(
        'empathetic_trader',
        [['active_listening'], ['validation'], ['problem_solving']]
    )
    
    print("\n\nIntegration test complete!")