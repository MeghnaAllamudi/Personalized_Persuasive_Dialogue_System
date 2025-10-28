import json
from collections import defaultdict
import random

# Map CaSiNo annotations to deescalation strategies
STRATEGY_MAPPING = {
    'showing-empathy': 'empathy',
    'self-disclosure': 'empathy',
    'emotional-appeal': 'empathy',
    'no-need': 'validation',
    'you-have-to': 'validation',
    'appeal-to-reasonableness': 'validation',
    'elicit-pref': 'active_listening',
    'uv-part': 'active_listening',
    'promote-coordination': 'problem_solving',
    'propose-a-deal': 'problem_solving',
    'task-related-inquiry': 'problem_solving',
}

def extract_examples():
    with open('data/casino_processed.json') as f:
        dialogues = json.load(f)
    
    example_bank = defaultdict(list)
    
    for dialogue in dialogues:
        turns = dialogue['turns']
        
        for i, turn in enumerate(turns):
            # Get previous turn for context
            prev_turn = turns[i-1] if i > 0 else None
            
            if 'annotations' in turn and prev_turn:
                for annotation in turn['annotations']:
                    strategy_name = annotation['name']
                    
                    # Map to our deescalation strategies
                    if strategy_name in STRATEGY_MAPPING:
                        deesc_strategy = STRATEGY_MAPPING[strategy_name]
                        
                        example_bank[deesc_strategy].append({
                            'user_message': prev_turn['text'],
                            'agent_response': turn['text'],
                            'strategy': deesc_strategy,
                            'original_annotation': strategy_name
                        })
    
    # Keep top 20 diverse examples per strategy
    for strategy in example_bank:
        examples = example_bank[strategy]
        # Simple diversity: shuffle and take first 20
        random.shuffle(examples)
        example_bank[strategy] = examples[:20]
    
    # Add manual examples for strategies not well-covered
    manual_examples = {
        'authority': [
            {
                'user_message': "I don't care about your rules!",
                'agent_response': "I understand your frustration. These policies are in place to ensure everyone's safety. Let's work within that framework to find a solution.",
                'strategy': 'authority'
            },
            {
                'user_message': "You can't tell me what to do!",
                'agent_response': "You're right that this is your choice. I'm here to explain the guidelines and help you make an informed decision that works for you.",
                'strategy': 'authority'
            }
        ]
    }
    
    for strategy, examples in manual_examples.items():
        if strategy not in example_bank:
            example_bank[strategy] = []
        example_bank[strategy].extend(examples)
    
    # Save
    with open('data/strategy_example_bank.json', 'w') as f:
        json.dump(dict(example_bank), f, indent=2)
    
    print(f"Example bank created:")
    for strategy, examples in example_bank.items():
        print(f"  {strategy}: {len(examples)} examples")
    
    return example_bank

if __name__ == '__main__':
    extract_examples()