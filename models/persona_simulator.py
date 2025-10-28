import os
import json
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import sys
sys.path.append('.')
from personas.persona_definitions import PERSONA_DEFINITIONS

load_dotenv()

class PersonaSimulator:
    """Simulate user responses based on personality type"""
    
    def __init__(self, persona_type, model="gpt-4o-mini", scenario=None):
        self.persona_type = persona_type
        self.persona_config = PERSONA_DEFINITIONS[persona_type]
        self.model = model
        self.conversation_history = []
        self.scenario = scenario or self._default_scenario()
        
        # Initialize API client
        if 'gpt' in model:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.provider = 'openai'
        elif 'claude' in model:
            self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.provider = 'anthropic'
    
    def get_initial_message(self):
        """
        Generate persona's initial message based on the scenario
        
        Returns:
            Persona's initial emotional/reactive message
        """
        prompt = self.persona_config['system_prompt']
        prompt += f"\n\nSITUATION:\n{self.scenario}\n"
        prompt += "\nThis is the beginning of the interaction. Express your initial feelings about this situation in 1-2 sentences, as this person would.\n"
        prompt += "You:"
        
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=100
            )
            return response.choices[0].message.content
        
        elif self.provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=0.8,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    def get_response(self, agent_message, strategy_used=None):
        """
        Generate persona's response to agent message
        
        Args:
            agent_message: What the deescalation agent said
            strategy_used: Optional list of strategies the agent used
        
        Returns:
            Persona's response string
        """
        prompt = self._build_persona_prompt(agent_message)
        
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=150
            )
            text = response.choices[0].message.content
        
        elif self.provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                temperature=0.8,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
        
        # Record in history
        self.conversation_history.append({
            'agent': agent_message,
            'user': text,
            'strategy': strategy_used
        })
        
        return text
    
    def _build_persona_prompt(self, agent_message):
        """Build prompt for persona simulation"""
        
        prompt = self.persona_config['system_prompt']
        
        prompt += f"\n\nSITUATION:\n{self.scenario}\n"
        
        if self.conversation_history:
            prompt += "\nCONVERSATION SO FAR:\n"
            for turn in self.conversation_history:
                prompt += f"Agent: {turn['agent']}\n"
                prompt += f"You: {turn['user']}\n"
        
        prompt += f"\nAgent: {agent_message}\n"
        prompt += "\nRespond as this person would. Keep your response to 1-3 sentences.\n"
        prompt += "You:"
        
        return prompt
    
    def _default_scenario(self):
        """Generate scenario based on persona type"""
        scenarios = {
            "aggressive": "You've been waiting at a checkpoint for over an hour. You have an important meeting and feel you're being treated unfairly. You're frustrated and angry.",
            "cooperative": "You're at a checkpoint and there seems to be confusion about your paperwork. You want to resolve this efficiently and fairly.",
            "anxious": "You're at a security checkpoint and you're worried something is wrong with your documentation. You're nervous about what might happen.",
            "stubborn": "You believe you have the right to pass through this checkpoint immediately. The rules being cited don't seem fair or applicable to your situation.",
            "diplomatic": "You're at a checkpoint and there appears to be a miscommunication about requirements. You want to resolve this professionally."
        }
        return scenarios.get(self.persona_type, "You're in a tense situation that needs to be resolved.")
    
    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []


# Test the simulator
if __name__ == '__main__':
    print("Testing persona simulators:\n")
    
    for persona_type in ['aggressive', 'cooperative', 'anxious']:
        print(f"\n{'='*80}")
        print(f"PERSONA: {persona_type.upper()}")
        print('='*80)
        
        simulator = PersonaSimulator(persona_type, model="gpt-4o-mini")
        
        # Simulate short interaction
        agent_messages = [
            "Hello, I understand you've been waiting. I'm here to help resolve this.",
            "I can see this is frustrating. Let me explain what we need to move forward.",
        ]
        
        for msg in agent_messages:
            print(f"\nAgent: {msg}")
            response = simulator.get_response(msg)
            print(f"{persona_type.title()}: {response}")