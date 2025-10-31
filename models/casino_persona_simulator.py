import os
import json
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import sys
sys.path.append('.')
from personas.casino_persona_definitions import CASINO_PERSONA_DEFINITIONS

load_dotenv()

class CasinoPersonaSimulator:
    """Simulate negotiator responses based on casino negotiation personality types"""
    
    def __init__(self, persona_type, model="gpt-4o-mini"):
        self.persona_type = persona_type
        self.persona_config = CASINO_PERSONA_DEFINITIONS[persona_type]
        self.model = model
        self.conversation_history = []
        
        # Initialize API client
        if 'gpt' in model:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.provider = 'openai'
        elif 'claude' in model:
            self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.provider = 'anthropic'
    
    def get_response(self, other_message):
        """
        Generate persona's response to other negotiator's message
        
        Args:
            other_message: What the other negotiator said
        
        Returns:
            Persona's response string
        """
        prompt = self._build_persona_prompt(other_message)
        
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,  # Higher temperature for more natural variation
                max_tokens=30  # Strict limit to force brevity like real humans
            )
            text = response.choices[0].message.content
        
        elif self.provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=30,  # Strict limit to force brevity like real humans
                temperature=0.9,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
        
        # Record in history
        self.conversation_history.append({
            'other': other_message,
            'you': text
        })
        
        return text
    
    def _build_persona_prompt(self, other_message):
        """Build prompt for persona simulation"""
        
        prompt = self.persona_config['system_prompt']
        
        if self.conversation_history:
            prompt += "\n\nCONVERSATION SO FAR:\n"
            for turn in self.conversation_history[-3:]:  # Keep last 3 turns for context
                prompt += f"Them: {turn['other']}\n"
                prompt += f"You: {turn['you']}\n"
        
        prompt += f"\nThem: {other_message}\n"
        prompt += "\nYour response (ONE short sentence, 10-20 words MAX, like texting):\n"
        prompt += "You:"
        
        return prompt
    
    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []


# Test the simulator
if __name__ == '__main__':
    print("Testing casino persona simulators:\n")
    
    test_messages = [
        "Hello! What supplies are you most interested in?",
        "I really need firewood for my kids to stay warm.",
        "How about I give you 2 firewood for 2 water and 1 food?"
    ]
    
    for persona_type in ['competitive_bargainer', 'empathetic_trader', 'strategic_negotiator']:
        print(f"\n{'='*80}")
        print(f"PERSONA: {persona_type.upper()}")
        print('='*80)
        
        simulator = CasinoPersonaSimulator(persona_type, model="gpt-4o-mini")
        
        for msg in test_messages:
            print(f"\nOther: {msg}")
            response = simulator.get_response(msg)
            print(f"You ({persona_type}): {response}")

