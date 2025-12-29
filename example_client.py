"""
Example client for the Personalized Persuasive Dialogue System API
Demonstrates both MAML-enhanced and baseline (random) conversations.
"""

import requests

BASE_URL = "http://10.0.0.86:8001"


def run_conversation(use_maml=True):
    """Run a sample conversation with or without MAML"""
    
    mode_label = "MAML-Enhanced" if use_maml else "Random Baseline"
    print(f"\n{'='*60}")
    print(f"  {mode_label} Conversation")
    print(f"{'='*60}\n")
    
    # Start conversation
    response = requests.post(f"{BASE_URL}/conversation/start", json={
        "mode": "user",
        "persona": "empathetic_trader",
        "initial_message": "Hi! I really need some firewood for my kids' camping trip.",
        "use_maml": use_maml
    })
    
    data = response.json()
    session_id = data['session_id']
    
    print(f"You: Hi! I really need some firewood for my kids' camping trip.")
    print(f"Agent: {data['response']}")
    print(f"  [Strategies: {data.get('strategies_used', 'N/A')}]\n")
    
    # Message 1
    response = requests.post(f"{BASE_URL}/conversation/message", json={
        "session_id": session_id,
        "message": "I have extra water bottles. Would you trade 2 firewood for 3 water?"
    })
    
    data = response.json()
    print(f"You: I have extra water bottles. Would you trade 2 firewood for 3 water?")
    print(f"Agent: {data['response']}")
    print(f"  [Strategies: {data.get('strategies_used', 'N/A')}]\n")
    
    # Message 2
    response = requests.post(f"{BASE_URL}/conversation/message", json={
        "session_id": session_id,
        "message": "That sounds fair! Let's do the trade. My kids will be so happy!"
    })
    
    data = response.json()
    print(f"You: That sounds fair! Let's do the trade. My kids will be so happy!")
    print(f"Agent: {data['response']}")
    print(f"  [Strategies: {data.get('strategies_used', 'N/A')}]")
    
    return session_id


def main():
    print("\n" + "="*60)
    print("  Personalized Persuasive Dialogue System - Example Client")
    print("="*60)
    
    # Run with MAML
    run_conversation(use_maml=True)
    
    # Run without MAML (random baseline)
    run_conversation(use_maml=False)
    
    print(f"\n{'='*60}")
    print("  Done! Compare the strategies used above.")
    print("  MAML selects strategies based on conversation context.")
    print("  Random baseline picks 1-2 strategies randomly each turn.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
