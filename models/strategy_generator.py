import os
import json
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class StrategyPromptedGenerator:
    """Generate deescalation responses using prompted LLMs"""
    
    def __init__(self, model="gpt-4o-mini", example_bank_path='data/strategy_example_bank.json'):
        self.model = model
        
        # Load example bank
        with open(example_bank_path) as f:
            self.example_bank = json.load(f)
        
        # Initialize API client
        if 'gpt' in model:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.provider = 'openai'
        elif 'claude' in model:
            self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.provider = 'anthropic'
    
    def generate_response(self, conversation_history, strategy_combination):
        """
        Generate response using specified strategies via few-shot prompting
        
        Args:
            conversation_history: List of dicts with 'user' and 'agent' keys
            strategy_combination: List of strategies to use (e.g., ['empathy', 'validation'])
        
        Returns:
            Generated response string
        """
        prompt = self._build_prompt(strategy_combination, conversation_history)
        
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content
        
        elif self.provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    def _build_prompt(self, strategies, conversation_history):
        """Build few-shot prompt with strategy examples"""
        
        STRATEGY_DESCRIPTIONS = {
            'empathy': "Acknowledge emotions and show understanding of their feelings",
            'validation': "Recognize their concerns as legitimate and reasonable",
            'active_listening': "Reflect back what they said to show you heard them",
            'problem_solving': "Propose collaborative solutions that address their needs",
            'authority': "Reference policies calmly while maintaining respect"
        }
        
        # Build prompt
        prompt = """You are a professional deescalation agent. Your goal is to reduce tension and find peaceful resolution through effective communication.

"""
        
        # Add strategy instructions
        prompt += f"STRATEGIES TO USE: {', '.join(strategies)}\n"
        for strategy in strategies:
            if strategy in STRATEGY_DESCRIPTIONS:
                prompt += f"- {strategy.upper()}: {STRATEGY_DESCRIPTIONS[strategy]}\n"
        
        # Add examples
        prompt += "\nEXAMPLES OF THESE STRATEGIES IN ACTION:\n\n"
        
        for strategy in strategies:
            if strategy in self.example_bank:
                # Use 2 examples per strategy
                for example in self.example_bank[strategy][:2]:
                    prompt += f"User: {example['user_message']}\n"
                    prompt += f"Agent: {example['agent_response']} [Strategy: {example['strategy']}]\n\n"
        
        # Add current conversation
        if conversation_history:
            prompt += "CURRENT CONVERSATION:\n"
            for turn in conversation_history:
                if 'user' in turn and turn['user']:
                    prompt += f"User: {turn['user']}\n"
                if 'agent' in turn and turn['agent']:
                    prompt += f"Agent: {turn['agent']}\n"
            
            # Get last user message
            last_user_message = conversation_history[-1].get('user', '')
            if last_user_message:
                prompt += f"\nUser: {last_user_message}\n\n"
        else:
            # First turn - no conversation history yet
            prompt += "\nThis is the beginning of the conversation.\n\n"
        
        prompt += f"Using the strategies [{', '.join(strategies)}], provide an appropriate deescalation response:\n"
        prompt += "Agent:"
        
        return prompt


# Test the generator
if __name__ == '__main__':
    generator = StrategyPromptedGenerator(model="gpt-4o-mini")
    
    # Test conversation
    conversation = [
        {"user": "This is completely unfair! I've been waiting for hours!", "agent": ""}
    ]
    
    # Test different strategy combinations
    print("Testing strategy-prompted generation:\n")
    
    strategies_to_test = [
        ['empathy'],
        ['validation'],
        ['empathy', 'validation'],
        ['empathy', 'problem_solving'],
        ['active_listening', 'validation']
    ]
    
    for strategies in strategies_to_test:
        print(f"\nStrategies: {strategies}")
        response = generator.generate_response(conversation, strategies)
        print(f"Response: {response}")
        print("-" * 80)