import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
import json
from datetime import datetime, timedelta
import random

class MentalHealthConversationDataset:
    """
    Generate synthetic mental health conversation dataset for AI chatbot training
    using SDV with CTGAN algorithm.
    """
    
    def __init__(self, num_samples=10000, seed=42):
        self.num_samples = num_samples
        self.seed = seed
        self.metadata = None
        self.synthesizer = None
        self.synthetic_data = None
        np.random.seed(seed)
        random.seed(seed)
    
    def create_base_dataset(self):
        """Create a base dataset with realistic mental health conversation patterns"""
        
        # Mental health categories and topics
        categories = [
            'depression', 'anxiety', 'stress', 'loneliness', 
            'self_esteem', 'relationships', 'work_stress', 
            'sleep_issues', 'grief', 'trauma', 'general_wellbeing'
        ]
        
        # Common user emotions and states
        emotions = [
            'sad', 'anxious', 'stressed', 'overwhelmed', 'lonely',
            'hopeless', 'confused', 'tired', 'nervous', 'panicked',
            'calm', 'neutral', 'slightly_positive', 'positive'
        ]
        
        # Severity levels
        severities = ['mild', 'moderate', 'severe', 'crisis']
        
        # Sample user messages with different emotional tones
        user_messages = [
            "I've been feeling really down lately",
            "I can't stop worrying about everything",
            "I feel so alone in this world",
            "My anxiety is making it hard to function",
            "I'm having trouble sleeping at night",
            "I don't feel motivated to do anything",
            "I keep having panic attacks",
            "I'm struggling with my self-esteem",
            "I feel overwhelmed by work/school",
            "I'm grieving a loss and it's hard to cope",
            "I want to improve my mental health",
            "I need someone to talk to",
            "I'm feeling stressed about relationships",
            "I think I might be depressed",
            "How can I manage my anxiety better?",
            "I need help with coping strategies",
            "I'm feeling a bit better today",
            "I want to work on my mental wellbeing",
            "I'm having negative thoughts",
            "I need support right now"
        ]
        
        # AI response templates (will be customized based on context)
        ai_response_templates = [
            "I understand you're feeling {emotion}. Let's talk about what's been going on.",
            "It sounds like you're dealing with a lot right now. Would you like to share more?",
            "I'm here to listen and support you. What's been on your mind lately?",
            "Thank you for sharing that with me. Let's explore some coping strategies together.",
            "I hear the {severity} in what you're describing. Have you considered {suggestion}?",
            "It's completely normal to feel {emotion} sometimes. Let's work through this together.",
            "I appreciate you reaching out. What's one small thing that might help right now?",
            "I'm sorry you're going through this. Would breathing exercises help in this moment?",
            "Let's break this down into smaller, manageable steps. What feels most urgent?",
            "Remember that it's okay to not be okay. What support do you need right now?"
        ]
        
        # Therapeutic techniques and suggestions
        techniques = [
            "deep breathing exercises", "mindfulness meditation", "journaling your thoughts",
            "progressive muscle relaxation", "challenging negative thoughts", 
            "setting small achievable goals", "reaching out to support networks",
            "practicing self-compassion", "creating a routine", "physical activity",
            "limiting screen time", "practicing gratitude", "seeking professional help",
            "using positive affirmations", "engaging in hobbies"
        ]
        
        # Generate synthetic conversations
        data = []
        for i in range(self.num_samples):
            # Random conversation parameters
            category = random.choice(categories)
            emotion = random.choice(emotions)
            severity = random.choice(severities)
            user_msg = random.choice(user_messages)
            
            # Customize AI response based on context
            ai_response = random.choice(ai_response_templates)
            if '{emotion}' in ai_response:
                ai_response = ai_response.format(emotion=emotion, severity=severity, 
                                               suggestion=random.choice(techniques))
            elif '{severity}' in ai_response:
                ai_response = ai_response.format(severity=severity, emotion=emotion,
                                               suggestion=random.choice(techniques))
            else:
                ai_response = ai_response.format(suggestion=random.choice(techniques))
            
            # Generate conversation metadata
            crisis_flag = severity == 'crisis'
            follow_up_needed = random.choice([True, False])
            
            conversation = {
                'conversation_id': f'conv_{i:05d}',
                'timestamp': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
                'user_message': user_msg,
                'ai_response': ai_response,
                'category': category,
                'detected_emotion': emotion,
                'severity_level': severity,
                'conversation_length': random.randint(2, 10),
                'user_demographic': random.choice(['18-25', '26-35', '36-45', '46-55', '56+']),
                'time_of_day': random.choice(['morning', 'afternoon', 'evening', 'night']),
                'crisis_flag': crisis_flag,  # Use boolean instead of integer
                'follow_up_needed': follow_up_needed,  # Use boolean instead of integer
                'therapeutic_approach': random.choice(['CBT', 'DBT', 'mindfulness', 'supportive', 'eclectic']),
                'response_quality_score': random.uniform(3.5, 5.0),
                'user_engagement_score': random.uniform(0.6, 1.0)
            }
            
            data.append(conversation)
        
        return pd.DataFrame(data)
    
    def create_metadata(self):
        """Create metadata for the dataset"""
        metadata = SingleTableMetadata()
        
        metadata.add_column('conversation_id', sdtype='id')
        metadata.add_column('timestamp', sdtype='datetime')
        metadata.add_column('user_message', sdtype='text')
        metadata.add_column('ai_response', sdtype='text')
        metadata.add_column('category', sdtype='categorical')
        metadata.add_column('detected_emotion', sdtype='categorical')
        metadata.add_column('severity_level', sdtype='categorical')
        metadata.add_column('conversation_length', sdtype='numerical')
        metadata.add_column('user_demographic', sdtype='categorical')
        metadata.add_column('time_of_day', sdtype='categorical')
        metadata.add_column('crisis_flag', sdtype='boolean')
        metadata.add_column('follow_up_needed', sdtype='boolean')
        metadata.add_column('therapeutic_approach', sdtype='categorical')
        metadata.add_column('response_quality_score', sdtype='numerical')
        metadata.add_column('user_engagement_score', sdtype='numerical')
        
        metadata.set_primary_key('conversation_id')
        
        # Add datetime format to avoid warning
        metadata.update_column(
            column_name='timestamp',
            datetime_format='%Y-%m-%dT%H:%M:%S.%f'
        )
        
        self.metadata = metadata
        return metadata
    
    def train_synthesizer(self, real_data, epochs=100):
        """Train the CTGAN synthesizer"""
        self.create_metadata()
        
        synthesizer = CTGANSynthesizer(
            self.metadata,
            epochs=epochs,
            verbose=True,
            cuda=False  # Set to True if you have GPU
        )
        
        synthesizer.fit(real_data)
        self.synthesizer = synthesizer
        return synthesizer
    
    def generate_synthetic_data(self, num_samples=None):
        """Generate synthetic data"""
        if num_samples is None:
            num_samples = self.num_samples
        
        if self.synthesizer is None:
            raise ValueError("Synthesizer not trained. Call train_synthesizer() first.")
        
        synthetic_data = self.synthesizer.sample(num_samples)
        self.synthetic_data = synthetic_data
        return synthetic_data
    
    def evaluate_quality(self, real_data):
        """Evaluate the quality of synthetic data"""
        if self.synthetic_data is None:
            raise ValueError("No synthetic data generated. Call generate_synthetic_data() first.")
        
        quality_report = evaluate_quality(
            real_data,
            self.synthetic_data,
            self.metadata
        )
        
        return quality_report
    
    def save_dataset(self, filename='mental_health_conversations.json'):
        """Save the dataset to a file"""
        if self.synthetic_data is None:
            raise ValueError("No synthetic data to save.")
        
        # Convert to dictionary format for better JSON structure
        dataset = self.synthetic_data.to_dict('records')
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {filename}")
    
    def load_dataset(self, filename):
        """Load dataset from file"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return pd.DataFrame(data)
    
    def create_rag_training_data(self):
        """Create formatted data for RAG training"""
        if self.synthetic_data is None:
            raise ValueError("No synthetic data available.")
        
        rag_data = []
        for _, row in self.synthetic_data.iterrows():
            rag_entry = {
                'input': row['user_message'],
                'output': row['ai_response'],
                'context': {
                    'category': row['category'],
                    'emotion': row['detected_emotion'],
                    'severity': row['severity_level'],
                    'demographic': row['user_demographic'],
                    'time_of_day': row['time_of_day'],
                    'crisis': bool(row['crisis_flag'])
                }
            }
            rag_data.append(rag_entry)
        
        return rag_data

# Alternative simplified approach without SDV for faster generation
def generate_simple_dataset(num_samples=5000, output_file='mental_health_conversations.json'):
    """Generate a simple dataset without SDV for quick testing"""
    
    # Mental health categories and topics
    categories = [
        'depression', 'anxiety', 'stress', 'loneliness', 
        'self_esteem', 'relationships', 'work_stress', 
        'sleep_issues', 'grief', 'trauma', 'general_wellbeing'
    ]
    
    # Common user emotions and states
    emotions = [
        'sad', 'anxious', 'stressed', 'overwhelmed', 'lonely',
        'hopeless', 'confused', 'tired', 'nervous', 'panicked',
        'calm', 'neutral', 'slightly_positive', 'positive'
    ]
    
    # Severity levels
    severities = ['mild', 'moderate', 'severe', 'crisis']
    
    # Sample user messages with different emotional tones
    user_messages = [
        "I've been feeling really down lately",
        "I can't stop worrying about everything",
        "I feel so alone in this world",
        "My anxiety is making it hard to function",
        "I'm having trouble sleeping at night",
        "I don't feel motivated to do anything",
        "I keep having panic attacks",
        "I'm struggling with my self-esteem",
        "I feel overwhelmed by work/school",
        "I'm grieving a loss and it's hard to cope",
        "I want to improve my mental health",
        "I need someone to talk to",
        "I'm feeling stressed about relationships",
        "I think I might be depressed",
        "How can I manage my anxiety better?",
        "I need help with coping strategies",
        "I'm feeling a bit better today",
        "I want to work on my mental wellbeing",
        "I'm having negative thoughts",
        "I need support right now"
    ]
    
    # AI response templates (will be customized based on context)
    ai_response_templates = [
        "I understand you're feeling {emotion}. Let's talk about what's been going on.",
        "It sounds like you're dealing with a lot right now. Would you like to share more?",
        "I'm here to listen and support you. What's been on your mind lately?",
        "Thank you for sharing that with me. Let's explore some coping strategies together.",
        "I hear the {severity} in what you're describing. Have you considered {suggestion}?",
        "It's completely normal to feel {emotion} sometimes. Let's work through this together.",
        "I appreciate you reaching out. What's one small thing that might help right now?",
        "I'm sorry you're going through this. Would breathing exercises help in this moment?",
        "Let's break this down into smaller, manageable steps. What feels most urgent?",
        "Remember that it's okay to not be okay. What support do you need right now?"
    ]
    
    # Therapeutic techniques and suggestions
    techniques = [
        "deep breathing exercises", "mindfulness meditation", "journaling your thoughts",
        "progressive muscle relaxation", "challenging negative thoughts", 
        "setting small achievable goals", "reaching out to support networks",
        "practicing self-compassion", "creating a routine", "physical activity",
        "limiting screen time", "practicing gratitude", "seeking professional help",
        "using positive affirmations", "engaging in hobbies"
    ]
    
    # Generate synthetic conversations
    data = []
    for i in range(num_samples):
        # Random conversation parameters
        category = random.choice(categories)
        emotion = random.choice(emotions)
        severity = random.choice(severities)
        user_msg = random.choice(user_messages)
        
        # Customize AI response based on context
        ai_response = random.choice(ai_response_templates)
        if '{emotion}' in ai_response:
            ai_response = ai_response.format(emotion=emotion, severity=severity, 
                                           suggestion=random.choice(techniques))
        elif '{severity}' in ai_response:
            ai_response = ai_response.format(severity=severity, emotion=emotion,
                                           suggestion=random.choice(techniques))
        else:
            ai_response = ai_response.format(suggestion=random.choice(techniques))
        
        # Generate conversation metadata
        crisis_flag = severity == 'crisis'
        follow_up_needed = random.choice([True, False])
        
        conversation = {
            'conversation_id': f'conv_{i:05d}',
            'timestamp': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
            'user_message': user_msg,
            'ai_response': ai_response,
            'category': category,
            'detected_emotion': emotion,
            'severity_level': severity,
            'conversation_length': random.randint(2, 10),
            'user_demographic': random.choice(['18-25', '26-35', '36-45', '46-55', '56+']),
            'time_of_day': random.choice(['morning', 'afternoon', 'evening', 'night']),
            'crisis_flag': crisis_flag,
            'follow_up_needed': follow_up_needed,
            'therapeutic_approach': random.choice(['CBT', 'DBT', 'mindfulness', 'supportive', 'eclectic']),
            'response_quality_score': round(random.uniform(3.5, 5.0), 2),
            'user_engagement_score': round(random.uniform(0.6, 1.0), 2)
        }
        
        data.append(conversation)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Simple dataset with {num_samples} samples saved to {output_file}")
    return pd.DataFrame(data)

# Example usage and dataset generation
def generate_mental_health_dataset():
    """Generate and save the mental health conversation dataset"""
    
    print("Creating mental health conversation dataset...")
    
    try:
        # Initialize dataset generator
        dataset_generator = MentalHealthConversationDataset(num_samples=5000)
        
        # Create base dataset
        print("Creating base dataset...")
        real_data = dataset_generator.create_base_dataset()
        print(f"Base dataset created with {len(real_data)} samples")
        
        # Train synthesizer
        print("Training CTGAN synthesizer...")
        synthesizer = dataset_generator.train_synthesizer(real_data, epochs=50)
        print("Synthesizer trained successfully!")
        
        # Generate synthetic data
        print("Generating synthetic data...")
        synthetic_data = dataset_generator.generate_synthetic_data()
        print(f"Synthetic data generated with {len(synthetic_data)} samples")
        
        # Evaluate quality
        print("Evaluating data quality...")
        quality_report = dataset_generator.evaluate_quality(real_data)
        print(f"Data quality score: {quality_report.get_score()}")
        
        # Save dataset
        print("Saving dataset...")
        dataset_generator.save_dataset('mental_health_chatbot_dataset.json')
        
        # Create RAG training data
        print("Creating RAG training data...")
        rag_data = dataset_generator.create_rag_training_data()
        
        with open('rag_training_data.json', 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, indent=2, ensure_ascii=False)
        
        print("RAG training data saved to rag_training_data.json")
        print("Dataset generation completed successfully!")
        
        return synthetic_data, rag_data
        
    except Exception as e:
        print(f"Error with SDV generation: {e}")
        print("Falling back to simple dataset generation...")
        return generate_simple_dataset(), []

if __name__ == "__main__":
    # Generate the dataset when run directly
    synthetic_data, rag_data = generate_mental_health_dataset()
    
    # Show sample data
    print("\nSample synthetic conversation:")
    if not synthetic_data.empty:
        print(synthetic_data[['user_message', 'ai_response', 'category', 'detected_emotion']].head(3).to_string())
    else:
        print("Using simple dataset format")