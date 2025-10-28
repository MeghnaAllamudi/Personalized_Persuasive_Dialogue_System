from datasets import load_dataset
import json
import os

# Download CaSiNo corpus
print("Downloading CaSiNo dataset...")
casino = load_dataset("casino")

# Explore structure
print(f"Total dialogues: {len(casino['train'])}")
sample = casino['train'][0]

print("\nSample dialogue structure:")
print(f"Keys: {sample.keys()}")
print(f"Dialogue turns: {len(sample['chat_logs'])}")
print(f"First turn: {sample['chat_logs'][0] if sample['chat_logs'] else 'No turns'}")
print(f"Annotations structure: {type(sample.get('annotations'))}")
if sample.get('annotations'):
    print(f"Sample annotations: {sample['annotations'][:2] if len(sample['annotations']) > 0 else sample['annotations']}")

# Extract strategy annotations (annotations are at dialogue level)
# Format: list of [utterance_text, comma_separated_strategies]
strategies_found = set()
for i in range(min(100, len(casino['train']))):
    dialogue = casino['train'][i]
    
    if not isinstance(dialogue, dict):
        print(f"Warning: Dialogue at index {i} is not a dict, it's {type(dialogue)}")
        continue
    
    # Check if annotations exist at dialogue level
    if 'annotations' in dialogue and dialogue['annotations']:
        annotations = dialogue['annotations']
        if isinstance(annotations, list):
            for annotation in annotations:
                # Each annotation is [utterance_text, strategies_string]
                if isinstance(annotation, list) and len(annotation) >= 2:
                    strategies_str = annotation[1]  # Second element contains strategies
                    if isinstance(strategies_str, str):
                        # Split comma-separated strategies
                        strategies = [s.strip() for s in strategies_str.split(',')]
                        strategies_found.update(strategies)

print(f"\nStrategies found: {strategies_found}")
print(f"Total unique strategies: {len(strategies_found)}")

# Save processed version
processed_data = []
for idx in range(len(casino['train'])):
    dialogue = casino['train'][idx]
    
    # Process annotations to extract strategies per turn
    turn_annotations = []
    if 'annotations' in dialogue and dialogue['annotations']:
        for annotation in dialogue['annotations']:
            if isinstance(annotation, list) and len(annotation) >= 2:
                turn_annotations.append({
                    'text': annotation[0],
                    'strategies': annotation[1].split(',') if isinstance(annotation[1], str) else []
                })
    
    processed_data.append({
        'dialogue_id': dialogue.get('dialogue_id', f'dialogue_{idx}'),
        'turns': dialogue['chat_logs'],
        'participant_info': dialogue['participant_info'],
        'annotations': turn_annotations
    })

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

with open('data/casino_processed.json', 'w') as f:
    json.dump(processed_data[:200], f, indent=2)  # Save first 200 for now

print("Saved to data/casino_processed.json")