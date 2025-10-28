from sentence_transformers import SentenceTransformer
import numpy as np
import torch

class ConversationStateEncoder:
    """Encode conversation history into fixed-size vector"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"Initialized encoder with {self.embedding_dim}-dim embeddings")
    
    def encode_conversation(self, conversation_history, max_turns=3):
        """
        Encode conversation history into state vector
        
        Args:
            conversation_history: List of dicts with 'agent' and 'user' keys
            max_turns: Maximum number of recent turns to include
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not conversation_history:
            # Empty conversation = zero vector
            return np.zeros(self.embedding_dim)
        
        # Take last N turns
        recent_turns = conversation_history[-max_turns:]
        
        # Concatenate into single text
        text_parts = []
        for turn in recent_turns:
            if 'agent' in turn and turn['agent']:
                text_parts.append(f"Agent: {turn['agent']}")
            if 'user' in turn and turn['user']:
                text_parts.append(f"User: {turn['user']}")
        
        full_text = " ".join(text_parts)
        
        # Encode
        embedding = self.encoder.encode(full_text, convert_to_numpy=True)
        
        return embedding
    
    def encode_batch(self, conversation_histories):
        """Encode multiple conversations"""
        embeddings = [self.encode_conversation(conv) for conv in conversation_histories]
        return np.array(embeddings)


# Test encoder
if __name__ == '__main__':
    encoder = ConversationStateEncoder()
    
    # Test conversation
    test_conversation = [
        {'agent': 'Hello, how can I help?', 'user': 'I am very frustrated!'},
        {'agent': 'I understand your frustration.', 'user': 'This is taking forever!'},
    ]
    
    embedding = encoder.encode_conversation(test_conversation)
    print(f"\nEncoded conversation to {len(embedding)}-dim vector")
    print(f"Embedding norm: {np.linalg.norm(embedding):.2f}")
    
    # Test that similar conversations have similar embeddings
    similar_conversation = [
        {'agent': 'Hi, what can I do for you?', 'user': 'I am really annoyed!'},
        {'agent': 'I see you are annoyed.', 'user': 'This is so slow!'},
    ]
    
    embedding2 = encoder.encode_conversation(similar_conversation)
    
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embedding], [embedding2])[0][0]
    print(f"Similarity to similar conversation: {similarity:.3f}")

