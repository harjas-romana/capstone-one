import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import requests
import json
import os
from typing import List, Dict, Optional, Tuple
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import time


os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"


class MentalHealthRAG:
    """
    RAG Layer for Mental Health Chatbot using ChromaDB and Groq API with Llama 3.3 70B.
    Handles mental health queries with context-aware responses and falls back to general
    queries for non-mental health topics.
    """
    
    def __init__(self, groq_api_key: str, chroma_db_path: str = "./chroma_mentalhealth_db"):
        self.groq_api_key = groq_api_key
        self.chroma_db_path = chroma_db_path
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_chroma_db()
        
        # Mental health keywords for query classification
        self.mental_health_keywords = [
            'mental', 'health', 'therapy', 'therapist', 'counseling', 'counselor',
            'depression', 'anxiety', 'stress', 'panic', 'attack', 'ptsd', 'trauma',
            'bipolar', 'ocd', 'adhd', 'autism', 'asperger', 'schizophrenia',
            'psychology', 'psychiatrist', 'psychiatric', 'medication', 'medicate',
            'emotional', 'feeling', 'feelings', 'sad', 'happy', 'angry', 'anger',
            'lonely', 'loneliness', 'isolated', 'isolation', 'suicide', 'suicidal',
            'selfharm', 'self-harm', 'self injury', 'cutting', 'therapy', 'therapeutic',
            'cope', 'coping', 'wellbeing', 'well-being', 'mindfulness', 'meditation',
            'mental illness', 'mental disorder', 'psychological', 'psychotherapy',
            'cognitive', 'behavioral', 'cbt', 'dbt', 'mental wellness', 'mental fitness',
            'burnout', 'exhausted', 'fatigue', 'sleep', 'insomnia', 'nightmare',
            'appetite', 'eating', 'disorder', 'anorexia', 'bulimia', 'binge',
            'addiction', 'alcohol', 'drug', 'substance', 'abuse', 'recovery',
            'grief', 'loss', 'mourning', 'bereavement', 'heartbreak', 'breakup',
            'relationship', 'marriage', 'couples', 'family', 'parenting', 'child',
            'teen', 'adolescent', 'elderly', 'aging', 'geriatric', 'memory',
            'dementia', 'alzheimer', 'focus', 'concentration', 'attention',
            'phobia', 'fear', 'worry', 'nervous', 'social anxiety', 'performance anxiety',
            'self-esteem', 'confidence', 'self-worth', 'body image', 'self-care',
            'resilience', 'coping skills', 'stress management', 'relaxation',
            'breathing', 'exercise', 'yoga', 'mind-body', 'holistic', 'alternative',
            'medication', 'antidepressant', 'ssri', 'snri', 'benzodiazepine', 'mood stabilizer',
            'side effects', 'withdrawal', 'therapy session', 'support group', 'peer support',
            'crisis', 'hotline', 'helpline', 'emergency', 'intervention', 'hospitalization',
            'inpatient', 'outpatient', 'treatment', 'rehabilitation', 'rehab'
        ]
        
        # General knowledge exclusion keywords
        self.general_knowledge_keywords = [
            'sports', 'movie', 'music', 'celebrity', 'politics', 'government', 'history',
            'science', 'technology', 'math', 'physics', 'chemistry', 'biology', 'geography',
            'travel', 'cooking', 'recipe', 'weather', 'news', 'entertainment', 'game',
            'business', 'economy', 'finance', 'stock', 'investment', 'shopping', 'product',
            'car', 'vehicle', 'computer', 'phone', 'device', 'hardware', 'software',
            'programming', 'code', 'language', 'english', 'spanish', 'french', 'german',
            'art', 'painting', 'literature', 'book', 'author', 'writer', 'poetry',
            'education', 'school', 'university', 'college', 'exam', 'test', 'homework',
            'job', 'career', 'employment', 'interview', 'resume', 'salary', 'workplace',
            'real estate', 'property', 'house', 'apartment', 'rent', 'mortgage',
            'fashion', 'clothing', 'beauty', 'cosmetics', 'healthcare', 'medical', 'doctor',
            'hospital', 'disease', 'illness', 'infection', 'virus', 'bacteria', 'vaccine',
            'nutrition', 'diet', 'vitamin', 'supplement', 'fitness', 'gym', 'workout'
        ]
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            # Fallback to Chroma's default embedding
            self.embedding_model = None
    
    def _initialize_chroma_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create Chroma client
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            
            # Create or get collection
            if self.embedding_model:
                # Use our own embedding function
                embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            else:
                # Use default embedding function
                embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            self.collection = self.chroma_client.get_or_create_collection(
                name="mental_health_knowledge",
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            
            print("ChromaDB initialized successfully")
            
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise
    
    def _is_mental_health_query(self, query: str) -> bool:
        """
        Determine if a query is related to mental health.
        Returns True for mental health queries, False for general knowledge queries.
        """
        query_lower = query.lower()
        
        # Check for mental health keywords
        mental_health_score = sum(1 for keyword in self.mental_health_keywords 
                                if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower))
        
        # Check for general knowledge keywords (to exclude)
        general_knowledge_score = sum(1 for keyword in self.general_knowledge_keywords 
                                    if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower))
        
        # If mental health keywords found and few general knowledge keywords, classify as mental health
        if mental_health_score > 0 and general_knowledge_score <= mental_health_score / 2:
            return True
        
        # Special cases that are always mental health
        if any(phrase in query_lower for phrase in [
            "how to help someone", "support someone", "coping with", "dealing with",
            "feeling depressed", "feeling anxious", "feeling stressed", "mental wellness",
            "emotional support", "therapy help", "counseling needed"
        ]):
            return True
            
        return False
    
    def add_knowledge_documents(self, documents: List[Dict[str, str]]):
        """
        Add mental health knowledge documents to the vector database.
        Each document should be a dict with 'text', 'source', and 'metadata' fields.
        """
        try:
            # Extract texts, metadatas, and ids
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                texts.append(doc['text'])
                
                # Convert list values to strings for ChromaDB compatibility
                metadata = {}
                for key, value in doc.get('metadata', {}).items():
                    if isinstance(value, list):
                        metadata[key] = ', '.join(value)  # Convert list to comma-separated string
                    else:
                        metadata[key] = value
                
                # Add basic metadata
                metadata.update({
                    'source': doc.get('source', 'unknown'),
                    'type': doc.get('type', 'general'),
                    'added_date': doc.get('added_date', ''),
                })
                
                metadatas.append(metadata)
                ids.append(f"doc_{i}_{int(time.time())}")
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Added {len(documents)} documents to knowledge base")
            return True
            
        except Exception as e:
            print(f"Error adding documents to knowledge base: {e}")
            return False
    
    def retrieve_relevant_context(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Retrieve relevant context from the knowledge base for a given query.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            contexts = []
            for i in range(len(results['documents'][0])):
                context = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None
                }
                contexts.append(context)
            
            return contexts
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def _format_rag_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Format the prompt for RAG-based response generation.
        """
        # Build context string
        context_str = ""
        for i, ctx in enumerate(contexts):
            context_str += f"Source [{i+1}]: {ctx['text']}\n\n"
    
    # Create the prompt
        prompt = f"""You are a compassionate mental health assistant from Group-33 (B.Tech CSE Cloud Computing & Automation). Provide supportive, evidence-based help.
        KNOWLEDGE SOURCES: {context_str}

        USER QUERY: {query}

        GUIDELINES:
        1. Be empathetic, validating, and non-judgmental
        2. Keep responses short, clear, and concise (2-3 paragraphs max)
        3. Use the provided knowledge sources to inform your response
        4. Only recommend professional help for serious medical concerns that require doctor intervention
        5. For crisis situations, provide appropriate hotline information
        6. Focus on practical coping strategies and recovery support
        7. Keep responses conversational and human-like

        RESPONSE:"""

        return prompt

    def _format_general_prompt(self, query: str) -> str:
        """
        Format the prompt for general knowledge queries.
        """
        prompt = f"""You are a mental health-focused AI assistant developed by Group-33 (B.Tech CSE Cloud Computing & Automation). 

        While I specialize in mental health support, I can provide brief answers to general questions.

        USER QUERY: {query}

        Please provide a single, concise response (1-2 sentences maximum) and mention that you are primarily a mental health assistant if the query is outside that scope and who developed you.

        RESPONSE:"""
        
        return prompt
    
    def call_groq_api(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Call the Groq API with the given prompt.
        """
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that provides accurate and compassionate responses."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "top_p": 1,
                "stream": False
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.Timeout:
            return "I'm sorry, the request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"I'm sorry, there was an error processing your request: {str(e)}"
        except Exception as e:
            return f"I'm sorry, an unexpected error occurred: {str(e)}"
    
    def generate_response(self, query: str, use_rag: bool = True) -> Dict[str, str]:
        """
        Generate a response to the user query, using RAG for mental health queries
        and direct API calls for general knowledge queries.
        """
        start_time = time.time()
        
        # Determine if this is a mental health query
        is_mental_health = self._is_mental_health_query(query)
        
        response_data = {
            "query": query,
            "is_mental_health": is_mental_health,
            "response": "",
            "contexts": [],
            "response_time": 0,
            "method": "rag" if (is_mental_health and use_rag) else "direct"
        }
        
        try:
            if is_mental_health and use_rag:
                # Mental health query with RAG
                contexts = self.retrieve_relevant_context(query)
                response_data["contexts"] = contexts
                
                if contexts:
                    prompt = self._format_rag_prompt(query, contexts)
                    response = self.call_groq_api(prompt)
                else:
                    # Fallback if no contexts found
                    prompt = self._format_general_prompt(query)
                    response = self.call_groq_api(prompt)
                    response_data["method"] = "direct_fallback"
                
            else:
                # General knowledge query or RAG disabled
                prompt = self._format_general_prompt(query)
                response = self.call_groq_api(prompt)
            
            response_data["response"] = response
            
        except Exception as e:
            response_data["response"] = f"I'm sorry, I encountered an error while processing your request: {str(e)}"
            response_data["method"] = "error"
        
        response_data["response_time"] = time.time() - start_time
        
        return response_data
    
    def search_similar_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Search for similar documents in the knowledge base.
        """
        return self.retrieve_relevant_context(query, n_results)
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the knowledge base collection.
        """
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "database_path": self.chroma_db_path
            }
        except:
            return {"error": "Unable to get collection statistics"}


# Example usage and initialization
def initialize_mental_health_knowledge_base(rag_system: MentalHealthRAG, sample_data: bool = True):
    """
    Initialize the mental health knowledge base with sample data.
    """
    if not sample_data:
        return
    
    # Sample mental health knowledge documents
    sample_documents = [
        {
            "text": "Depression is a common mental health disorder characterized by persistent sadness, loss of interest or pleasure, feelings of guilt or low self-worth, disturbed sleep or appetite, low energy, and poor concentration. It can be effectively treated with therapies like cognitive behavioral therapy (CBT), interpersonal therapy (IPT), and antidepressant medication.",
            "source": "World Health Organization",
            "type": "condition_overview",
            "metadata": {
                "condition": "depression",
                "severity": "general",
                "treatment_approaches": "therapy, medication"  # Converted from list to string
            }
        },
        {
            "text": "Anxiety disorders involve excessive fear and worry that are difficult to control and impact daily functioning. Common types include generalized anxiety disorder, panic disorder, social anxiety disorder, and specific phobias. Treatment often includes cognitive behavioral therapy, exposure therapy, relaxation techniques, and sometimes medication.",
            "source": "American Psychological Association",
            "type": "condition_overview",
            "metadata": {
                "condition": "anxiety",
                "severity": "general",
                "treatment_approaches": "CBT, exposure therapy, relaxation, medication"  # Converted from list to string
            }
        },
        {
            "text": "Cognitive Behavioral Therapy (CBT) is a widely used therapeutic approach that helps individuals identify and change negative thought patterns and behaviors. It's effective for depression, anxiety disorders, eating disorders, and other mental health conditions. CBT typically involves structured sessions where patients learn practical skills to manage their symptoms.",
            "source": "National Institute of Mental Health",
            "type": "treatment_method",
            "metadata": {
                "therapy_type": "CBT",
                "conditions": "depression, anxiety, eating disorders",  # Converted from list to string
                "evidence_level": "high"
            }
        },
        {
            "text": "If you're experiencing a mental health crisis, please reach out to emergency services or a crisis hotline immediately. In the US, you can call or text 988 to reach the Suicide & Crisis Lifeline, which provides free, confidential support 24/7. You're not alone, and help is available.",
            "source": "988 Suicide & Crisis Lifeline",
            "type": "crisis_resource",
            "metadata": {
                "resource_type": "crisis_support",
                "availability": "24/7",
                "contact": "988"
            }
        },
        {
            "text": "Mindfulness meditation involves paying attention to the present moment without judgment. Regular practice can reduce stress, improve emotional regulation, and enhance overall wellbeing. Start with just 5-10 minutes daily, focusing on your breath and gently bringing your attention back when it wanders.",
            "source": "Mindfulness-Based Stress Reduction Program",
            "type": "coping_skill",
            "metadata": {
                "skill_type": "mindfulness",
                "benefits": "stress reduction, emotional regulation, wellbeing",  # Converted from list to string
                "practice_time": "5-10 minutes daily"
            }
        },
        {
            "text": "Self-care is essential for mental health maintenance. This includes adequate sleep, regular physical activity, healthy nutrition, social connections, and activities you enjoy. Establishing a routine that incorporates these elements can significantly improve resilience to stress and overall mental wellbeing.",
            "source": "Mental Health America",
            "type": "prevention",
            "metadata": {
                "category": "self_care",
                "components": "sleep, exercise, nutrition, social, enjoyment"  # Converted from list to string
            }
        }
    ]
    
    # Add sample documents to the knowledge base
    success = rag_system.add_knowledge_documents(sample_documents)
    if success:
        print("Sample mental health knowledge base initialized")
    else:
        print("Failed to initialize sample mental health knowledge base")


if __name__ == "__main__":
    # Example usage
    import os
    
    # Get Groq API key from environment variable
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Please set GROQ_API_KEY environment variable")
        exit(1)
    
    # Initialize RAG system
    rag_system = MentalHealthRAG(groq_api_key=groq_api_key)
    
    # Initialize with sample data
    initialize_mental_health_knowledge_base(rag_system, sample_data=True)
    
    # test_queries = [
    #     "I've been feeling really depressed lately, what can I do?",
    #     "How does cognitive behavioral therapy work for anxiety?",
    #     "What's the capital of France?",
    #     "I'm having panic attacks and don't know how to cope",
    #     "Tell me about the latest football game"
    # ]
    
    # for query in test_queries:
    #     print(f"\n{'='*50}")
    #     print(f"QUERY: {query}")
        
    #     response = rag_system.generate_response(query)
        
    #     print(f"CLASSIFIED AS: {'Mental Health' if response['is_mental_health'] else 'General Knowledge'}")
    #     print(f"RESPONSE METHOD: {response['method']}")
    #     print(f"RESPONSE TIME: {response['response_time']:.2f}s")
    #     print(f"RESPONSE: {response['response']}")
        
    #     if response['contexts']:
    #         print(f"CONTEXTS USED: {len(response['contexts'])}")