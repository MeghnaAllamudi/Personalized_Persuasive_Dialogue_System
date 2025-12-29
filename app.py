"""
Flask API for Personalized Persuasive Dialogue System

Run with: python app.py
Access locally: http://localhost:5000
Access on network: http://<your-local-ip>:5000

Endpoints:
- POST /conversation/start - Start a new conversation
- POST /conversation/message - Send a message and get a response
- GET /conversation/<session_id> - Get conversation history
- GET /personas - List available personas
- GET /strategies - List available strategies
"""

import os
import json
import uuid
import random
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Import our dialogue system components
from models.strategy_generator import StrategyPromptedGenerator
from models.persona_simulator import CasinoPersonaSimulator
from models.state_encoder import ConversationStateEncoder
from models.strategy_selector import StrategySelector
import torch
import numpy as np

load_dotenv()

app = Flask(__name__)

# Store active conversations in memory
conversations = {}

# Initialize components
print("Initializing dialogue system components...")
strategy_generator = StrategyPromptedGenerator(model="gpt-4o-mini")
state_encoder = ConversationStateEncoder()

# Load MAML-trained strategy selector if available
strategy_selector = None
MAML_MODEL_PATH = 'results/maml_model.pt'
if os.path.exists(MAML_MODEL_PATH):
    print("Loading MAML-trained strategy selector...")
    strategy_selector = StrategySelector(state_dim=384, num_strategies=5)
    strategy_selector.load_state_dict(torch.load(MAML_MODEL_PATH, map_location='cpu'))
    strategy_selector.eval()
    print("MAML model loaded successfully!")
else:
    print("No MAML model found. Using default strategies.")

# Strategy mapping
STRATEGY_NAMES = ['empathy', 'validation', 'active_listening', 'problem_solving', 'authority']

# Available personas
AVAILABLE_PERSONAS = [
    'competitive_bargainer',
    'empathetic_trader', 
    'strategic_negotiator',
    'flexible_collaborator',
    'assertive_claimer'
]


def select_strategies(conversation_history, manual_strategies=None, use_maml=True):
    """Select strategies using MAML model or defaults"""
    if manual_strategies:
        return manual_strategies
    
    if use_maml and strategy_selector is not None:
        # Use MAML-trained model
        state = state_encoder.encode_conversation(conversation_history)
        selected_indices, probs = strategy_selector.select_strategies(state, threshold=0.4)
        selected = [STRATEGY_NAMES[i] for i in selected_indices]
        return selected
    else:
        # Random baseline - pick 1-2 random strategies
        num_strategies = random.randint(1, 2)
        return random.sample(STRATEGY_NAMES, num_strategies)


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API info"""
    return jsonify({
        'name': 'Personalized Persuasive Dialogue System API',
        'version': '1.0',
        'endpoints': {
            'POST /conversation/start': 'Start a new conversation',
            'POST /conversation/message': 'Send a message and get response',
            'GET /conversation/<session_id>': 'Get conversation history',
            'GET /personas': 'List available personas',
            'GET /strategies': 'List available strategies'
        },
        'maml_enabled': strategy_selector is not None
    })


@app.route('/personas', methods=['GET'])
def list_personas():
    """List all available personas"""
    return jsonify({
        'personas': AVAILABLE_PERSONAS,
        'descriptions': {
            'competitive_bargainer': 'Strategic, maximizes their share, competitive but not rude',
            'empathetic_trader': 'Uses personal stories and emotional appeals, warm and relatable',
            'strategic_negotiator': 'Logical, clear priorities, focuses on efficient deals',
            'flexible_collaborator': 'Cooperative, friendly, willing to adjust, uses positive language',
            'assertive_claimer': 'States needs directly, firm about requirements, less flexible'
        }
    })


@app.route('/strategies', methods=['GET'])
def list_strategies():
    """List all available persuasion strategies"""
    return jsonify({
        'strategies': STRATEGY_NAMES,
        'descriptions': {
            'empathy': 'Acknowledge emotions and show understanding of their feelings',
            'validation': 'Recognize their concerns as legitimate and reasonable',
            'active_listening': 'Reflect back what they said to show you heard them',
            'problem_solving': 'Propose collaborative solutions that address their needs',
            'authority': 'Reference policies calmly while maintaining respect'
        }
    })


@app.route('/conversation/start', methods=['POST'])
def start_conversation():
    """
    Start a new conversation
    
    Request body:
    {
        "persona": "empathetic_trader",  // optional, simulates user with this persona
        "mode": "agent" | "user",        // "agent" = you are the negotiator, "user" = you chat with agent
        "initial_message": "Hello!",     // optional opening message
        "use_maml": true                 // optional, set to false to disable MAML (uses random strategies)
    }
    
    Response:
    {
        "session_id": "uuid",
        "mode": "agent",
        "persona": "empathetic_trader",
        "response": "response from system",
        "strategies_used": ["empathy", "validation"],
        "use_maml": true
    }
    """
    data = request.get_json() or {}
    
    session_id = str(uuid.uuid4())
    mode = data.get('mode', 'agent')  # Default: you are the persuasive agent
    persona = data.get('persona', 'empathetic_trader')
    initial_message = data.get('initial_message')
    use_maml = data.get('use_maml', True)  # Default: use MAML if available
    
    if persona not in AVAILABLE_PERSONAS:
        return jsonify({'error': f'Invalid persona. Choose from: {AVAILABLE_PERSONAS}'}), 400
    
    # Initialize conversation state
    conversations[session_id] = {
        'mode': mode,
        'persona': persona,
        'history': [],
        'use_maml': use_maml,
        'persona_simulator': CasinoPersonaSimulator(persona, model="gpt-4o-mini") if mode == 'agent' else None
    }
    
    response_data = {
        'session_id': session_id,
        'mode': mode,
        'persona': persona,
        'response': None,
        'strategies_used': None,
        'use_maml': use_maml
    }
    
    if mode == 'agent':
        # You are the agent - persona simulator plays the user
        # Generate persona's opening message
        if initial_message:
            # Use provided initial message as user's first message
            conversations[session_id]['history'].append({
                'role': 'user',
                'content': initial_message
            })
            response_data['response'] = initial_message
            response_data['awaiting'] = 'agent_response'
        else:
            # Generate persona's opening
            opening = "Hi! I'm looking to trade some camping supplies. What do you need?"
            conversations[session_id]['history'].append({
                'role': 'user', 
                'content': opening
            })
            response_data['response'] = opening
            response_data['awaiting'] = 'agent_response'
    else:
        # You are the user - system plays the persuasive agent
        if initial_message:
            # Process your message and generate agent response
            conversations[session_id]['history'].append({
                'role': 'user',
                'content': initial_message
            })
            
            # Select strategies and generate response
            conv_history = [{'user': initial_message, 'agent': ''}]
            strategies = select_strategies(conv_history, use_maml=use_maml)
            agent_response = strategy_generator.generate_response(conv_history, strategies)
            
            conversations[session_id]['history'].append({
                'role': 'agent',
                'content': agent_response,
                'strategies': strategies
            })
            
            response_data['response'] = agent_response
            response_data['strategies_used'] = strategies
            response_data['awaiting'] = 'user_message'
        else:
            # Agent opens the conversation
            strategies = select_strategies([], use_maml=use_maml)
            opening = strategy_generator.generate_response([], strategies)
            
            conversations[session_id]['history'].append({
                'role': 'agent',
                'content': opening,
                'strategies': strategies
            })
            
            response_data['response'] = opening
            response_data['strategies_used'] = strategies
            response_data['awaiting'] = 'user_message'
    
    return jsonify(response_data)


@app.route('/conversation/message', methods=['POST'])
def send_message():
    """
    Send a message in an ongoing conversation
    
    Request body:
    {
        "session_id": "uuid",
        "message": "Your message here",
        "strategies": ["empathy", "validation"]  // optional, override strategy selection
    }
    
    Response:
    {
        "session_id": "uuid",
        "your_message": "what you sent",
        "response": "system response",
        "strategies_used": ["empathy", "validation"],
        "turn": 3
    }
    """
    data = request.get_json()
    
    if not data or 'session_id' not in data or 'message' not in data:
        return jsonify({'error': 'Missing session_id or message'}), 400
    
    session_id = data['session_id']
    message = data['message']
    manual_strategies = data.get('strategies')
    
    if session_id not in conversations:
        return jsonify({'error': 'Session not found. Start a new conversation.'}), 404
    
    conv = conversations[session_id]
    mode = conv['mode']
    
    if mode == 'agent':
        # You are the agent - your message is the agent's response
        # Add your (agent) message
        conv['history'].append({
            'role': 'agent',
            'content': message,
            'strategies': manual_strategies or []
        })
        
        # Build conversation history for persona simulator
        formatted_history = []
        for h in conv['history']:
            if h['role'] == 'user':
                formatted_history.append({'user': h['content'], 'agent': ''})
            elif h['role'] == 'agent' and formatted_history:
                formatted_history[-1]['agent'] = h['content']
        
        # Generate persona's response
        persona_sim = conv['persona_simulator']
        persona_response = persona_sim.get_response(message)
        
        conv['history'].append({
            'role': 'user',
            'content': persona_response
        })
        
        return jsonify({
            'session_id': session_id,
            'your_message': message,
            'response': persona_response,
            'persona': conv['persona'],
            'turn': len([h for h in conv['history'] if h['role'] == 'agent'])
        })
    
    else:
        # You are the user - system generates agent response
        conv['history'].append({
            'role': 'user',
            'content': message
        })
        
        # Build conversation history for strategy selection
        formatted_history = []
        for i in range(0, len(conv['history']), 2):
            user_msg = conv['history'][i]['content'] if i < len(conv['history']) else ''
            agent_msg = conv['history'][i+1]['content'] if i+1 < len(conv['history']) else ''
            formatted_history.append({'user': user_msg, 'agent': agent_msg})
        
        # Add current user message
        formatted_history.append({'user': message, 'agent': ''})
        
        # Select strategies (use MAML based on session setting)
        use_maml = conv.get('use_maml', True)
        strategies = select_strategies(formatted_history, manual_strategies, use_maml=use_maml)
        
        # Generate agent response
        agent_response = strategy_generator.generate_response(formatted_history, strategies)
        
        conv['history'].append({
            'role': 'agent',
            'content': agent_response,
            'strategies': strategies
        })
        
        return jsonify({
            'session_id': session_id,
            'your_message': message,
            'response': agent_response,
            'strategies_used': strategies,
            'use_maml': use_maml,
            'turn': len([h for h in conv['history'] if h['role'] == 'agent'])
        })


@app.route('/conversation/<session_id>', methods=['GET'])
def get_conversation(session_id):
    """Get full conversation history"""
    if session_id not in conversations:
        return jsonify({'error': 'Session not found'}), 404
    
    conv = conversations[session_id]
    return jsonify({
        'session_id': session_id,
        'mode': conv['mode'],
        'persona': conv['persona'],
        'history': conv['history'],
        'turns': len([h for h in conv['history'] if h['role'] == 'agent'])
    })


@app.route('/conversation/<session_id>', methods=['DELETE'])
def end_conversation(session_id):
    """End and delete a conversation"""
    if session_id in conversations:
        del conversations[session_id]
        return jsonify({'message': 'Conversation ended', 'session_id': session_id})
    return jsonify({'error': 'Session not found'}), 404


if __name__ == '__main__':
    import socket
    
    # Get local IP for network access
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*60)
    print("🚀 Personalized Persuasive Dialogue System API")
    print("="*60)
    print(f"\n📍 Local access:   http://localhost:8001")
    print(f"📍 Network access: http://{local_ip}:8001")
    print("\n📚 Endpoints:")
    print("   POST /conversation/start   - Start new conversation")
    print("   POST /conversation/message - Send message")
    print("   GET  /conversation/<id>    - Get history")
    print("   GET  /personas             - List personas")
    print("   GET  /strategies           - List strategies")
    print("="*60 + "\n")
    
    # Run on all interfaces so it's accessible on the network
    app.run(host='0.0.0.0', port=8001, debug=True)

