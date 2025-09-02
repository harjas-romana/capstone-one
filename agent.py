import os
import json
import time
from datetime import datetime
from typing import List, Dict
import readline

from rag import MentalHealthRAG

class MentalHealthAgent:

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.rag_system = MentalHealthRAG(groq_api_key)
        self.conversation_history = []
        self.current_session_id = None
        self.user_name = "User"
        
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):

        try:
            sample_documents = [
                {
                    "text": "Depression is treatable with therapy (CBT, IPT) and medication. Symptoms include persistent sadness, loss of interest, sleep/appetite changes, low energy, and poor concentration.",
                    "source": "WHO",
                    "type": "condition",
                    "metadata": {"condition": "depression", "severity": "general"}
                },
                {
                    "text": "Anxiety disorders involve excessive fear/worry. Treatment includes CBT, exposure therapy, relaxation techniques, and sometimes medication.",
                    "source": "APA",
                    "type": "condition",
                    "metadata": {"condition": "anxiety", "severity": "general"}
                },
                {
                    "text": "CBT helps identify and change negative thought patterns. Effective for depression, anxiety, eating disorders, and other mental health conditions.",
                    "source": "NIMH",
                    "type": "treatment",
                    "metadata": {"therapy": "CBT", "conditions": "depression, anxiety"}
                },
                {
                    "text": "For mental health crisis, call/text 988 (Suicide & Crisis Lifeline) for free, confidential support 24/7.",
                    "source": "988 Lifeline",
                    "type": "resource",
                    "metadata": {"resource": "crisis", "availability": "24/7"}
                },
                {
                    "text": "Mindfulness meditation reduces stress and improves emotional regulation. Practice 5-10 minutes daily focusing on breath.",
                    "source": "MBSR",
                    "type": "technique",
                    "metadata": {"technique": "mindfulness", "duration": "5-10min"}
                },
                {
                    "text": "Self-care includes adequate sleep, exercise, nutrition, social connections, and enjoyable activities for mental wellbeing.",
                    "source": "MHA",
                    "type": "prevention",
                    "metadata": {"category": "self_care", "components": "sleep, exercise, nutrition"}
                }
            ]
            
            self.rag_system.add_knowledge_documents(sample_documents)
            print("✓ Knowledge base initialized")
            
        except Exception as e:
            print(f"Could not initialize knowledge base: {e}")
    
    def start_new_chat(self):
        self.current_session_id = f"session_{int(time.time())}"
        self.conversation_history = []
        
        print("\n" + "="*60)
        print("NEW CHAT SESSION STARTED")
        print("="*60)
        print("Hi there! I'm your mental health assistant.")
        print("How can I support you today?")
        print("Type '/help' for available commands")
        print("="*60)
        
        self._add_to_history("system", "Hi there! I'm your mental health assistant. How can I support you today?")
    
    def _add_to_history(self, role: str, message: str):
        timestamp = datetime.now().isoformat()
        self.conversation_history.append({
            "role": role,
            "message": message,
            "timestamp": timestamp,
            "session_id": self.current_session_id
        })
    
    def _get_conversation_context(self, max_messages: int = 10) -> str:
        if not self.conversation_history:
            return ""
        
        recent_messages = [msg for msg in self.conversation_history[-max_messages:] 
                          if msg['role'] in ['user', 'assistant']]
        
        context = "Recent conversation context:\n"
        for msg in recent_messages:
            speaker = "USER" if msg['role'] == 'user' else "ASSISTANT"
            context += f"{speaker}: {msg['message']}\n"
        
        return context
    
    def process_message(self, user_input: str) -> str:
        if not user_input.strip():
            return "Please type something to chat with me."
        
        self._add_to_history("user", user_input)
        
        conversation_context = self._get_conversation_context()
        
        contextual_query = f"{conversation_context}\nCurrent query: {user_input}" if conversation_context else user_input
        
        start_time = time.time()
        response_data = self.rag_system.generate_response(contextual_query)
        response_time = time.time() - start_time
        
        response = response_data['response']
        
        self._add_to_history("assistant", response)
        
        return response
    
    def show_help(self):
        help_text = """
AVAILABLE COMMANDS:
/help       - Show this help message
/new        - Start a new chat session
/history    - Show conversation history
/clear      - Clear current conversation
/save       - Save conversation to file
/exit       - Exit the program
/about      - Show information about this agent
/stats      - Show knowledge base statistics
        """
        print(help_text)
    
    def show_history(self):
        if not self.conversation_history:
            print("No conversation history yet.")
            return
        
        print("\n" + "="*60)
        print("CONVERSATION HISTORY")
        print("="*60)
        
        for i, msg in enumerate(self.conversation_history, 1):
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime("%H:%M:%S")
            prefix = "👤 YOU: " if msg['role'] == 'user' else "🤖 AI: "
            print(f"{i:2d}. [{timestamp}] {prefix}{msg['message']}")
    
    def clear_history(self):
        self.conversation_history = []
        print("✓ Conversation history cleared")
    
    def save_conversation(self, filename: str = None):
        if not self.conversation_history:
            print("No conversation to save.")
            return
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        try:
            data = {
                "session_id": self.current_session_id,
                "created_at": datetime.now().isoformat(),
                "messages": self.conversation_history
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Conversation saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
    
    def show_about(self):
        about_text = """
🤖 MENTAL HEALTH CHATBOT AGENT
──────────────────────────────
Developed by: Group-33, B.Tech CSE Cloud Computing & Automation
Capstone Project: AI Mental Health Assistant
Team: Harjas, Divyam, Sonia, Manini, Jiyaa

Features:
• RAG-powered mental health support
• Conversation memory and context awareness
• Evidence-based information from trusted sources
• Crisis resource information
• General knowledge capabilities

Technology Stack:
• Groq API with Llama 3.3 70B Versatile
• ChromaDB vector database
• Sentence Transformers for embeddings
• Custom RAG layer for mental health
        """
        print(about_text)
    
    def show_stats(self):
        stats = self.rag_system.get_collection_stats()
        print("\n📊 KNOWLEDGE BASE STATISTICS")
        print("="*30)
        print(f"Documents: {stats.get('document_count', 'N/A')}")
        print(f"Database: {stats.get('database_path', 'N/A')}")
        print(f"Session ID: {self.current_session_id}")
        print(f"Messages in memory: {len(self.conversation_history)}")
    
    def run(self):
        print("🚀 Initializing Mental Health Chatbot Agent...")
        
        self.start_new_chat()
        
        while True:
            try:
                user_input = input("\n👤 YOU: ").strip()
                
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command in ['exit', 'quit']:
                        print("👋 Goodbye! Take care of yourself.")
                        break
                    
                    elif command in ['new', 'reset']:
                        self.start_new_chat()
                        continue
                    
                    elif command in ['help', '?']:
                        self.show_help()
                        continue
                    
                    elif command in ['history', 'hist']:
                        self.show_history()
                        continue
                    
                    elif command in ['clear', 'cls']:
                        self.clear_history()
                        continue
                    
                    elif command in ['save', 'export']:
                        self.save_conversation()
                        continue
                    
                    elif command in ['about', 'info']:
                        self.show_about()
                        continue
                    
                    elif command in ['stats', 'status']:
                        self.show_stats()
                        continue
                    
                    else:
                        print("Unknown command. Type /help for available commands.")
                        continue
                
                print("\n🤖 AI: ", end="", flush=True)
                
                response = self.process_message(user_input)

                for char in response:
                    print(char, end="", flush=True)
                    time.sleep(0.001)
                print()
                
            except KeyboardInterrupt:
                print("\n\n Interrupted. Type /exit to quit or continue chatting.")
                continue
            
            except Exception as e:
                print(f"\n Error: {e}")
                print("Please try again or type /new to start a fresh conversation.")


def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Please set GROQ_API_KEY environment variable")
        print("💡 Run: export GROQ_API_KEY=your_api_key_here")
        return
    
    agent = MentalHealthAgent(groq_api_key)
    agent.run()


if __name__ == "__main__":
    main()