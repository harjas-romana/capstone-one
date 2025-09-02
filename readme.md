# Mental Health Chatbot 🤖💚

A RAG-powered mental health assistant built with Groq API, Llama 3.3 70B, and ChromaDB. Provides empathetic, context-aware responses for mental health support while handling general queries.

## 🚀 Features

- **RAG Architecture**: Retrieval Augmented Generation for evidence-based mental health responses
- **Groq API Integration**: Ultra-fast inference with Llama 3.3 70B model
- **ChromaDB Vector Store**: Efficient knowledge retrieval and storage
- **Conversation Memory**: Maintains context across chat sessions
- **Dual Mode**: Mental health focus with general knowledge fallback
- **FastAPI Server**: Robust REST API with comprehensive logging
- **CLI Interface**: Interactive command-line chat interface

## 📁 Project Structure

```
mental-health-chatbot/
├── rag_layer.py          # RAG system with ChromaDB integration
├── agent.py             # CLI chat interface with memory
├── server.py            # FastAPI REST server
├── dataset.py           # Synthetic data generation (SDV)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd mental-health-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit the `.env` file and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🔑 Getting Groq API Key

1. Go to [Groq Console](https://console.groq.com/)
2. Sign up/login to your account
3. Navigate to API Keys section
4. Generate a new API key
5. Copy the key to your `.env` file

## 🚀 Usage

### Option 1: CLI Chat Interface
```bash
python agent.py
```

**Available Commands:**
- `/help` - Show help menu
- `/new` - Start new chat session
- `/history` - View conversation history
- `/save` - Save conversation to file
- `/exit` - Exit the program

### Option 2: FastAPI Server
```bash
python server.py
```

**API Endpoints:**
- `GET /` - Server information
- `GET /health` - Health check
- `POST /chat` - Send message to chatbot
- `GET /sessions` - List active sessions
- `GET /docs` - Interactive API documentation

### Option 3: Direct API Usage
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "I need help with anxiety"}'
```

## 🧠 How It Works

1. **Query Classification**: Determines if query is mental health-related
2. **RAG Retrieval**: For mental health queries, searches ChromaDB knowledge base
3. **Contextual Response**: Generates responses using retrieved context + Llama 3.3
4. **General Fallback**: For non-mental health queries, uses direct API calls
5. **Session Management**: Maintains conversation memory across interactions

## 📊 Knowledge Base

The system includes pre-loaded mental health knowledge about:
- Depression and anxiety disorders
- Cognitive Behavioral Therapy (CBT)
- Crisis resources and hotlines
- Mindfulness and coping techniques
- Self-care strategies

## 🔧 Configuration

### Environment Variables
```env
GROQ_API_KEY=your_groq_api_key_here
CHROMA_DB_PATH=./chroma_mentalhealth_db
LOG_LEVEL=INFO
```

### Custom Knowledge Base
Add your own mental health documents by modifying the sample data in:
- `rag_layer.py` - `initialize_mental_health_knowledge_base()`
- `server.py` - startup event

## 📋 API Documentation

Once the server is running, access interactive documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🐛 Troubleshooting

### Common Issues

1. **GROQ_API_KEY not set**
   ```bash
   export GROQ_API_KEY=your_actual_key_here
   ```

2. **Port already in use**
   ```bash
   # Change port in server.py or use:
   python server.py --port 8080
   ```

3. **ChromaDB errors**
   ```bash
   # Try resetting the database
   rm -rf chroma_mentalhealth_db/
   ```

### Logs
- Application logs: `mental_health_server.log`
- Access logs: Console output when running server

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Style
```bash
# Format code
black *.py

# Check linting
flake8 *.py
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This is a capstone project developed for educational purposes. It is not a substitute for professional mental health care. For serious mental health concerns, please contact licensed professionals or emergency services.

**Crisis Resources:**
- National Suicide Prevention Lifeline: 988 or 1-800-273-8255
- Crisis Text Line: Text "HOME" to 741741
- Emergency Services: 911

## 👥 Developers

**Group 33 - B.Tech CSE Cloud Computing & Automation**
- Harjas Singh
- [Other team members]

**Capstone Project** - Academic Year 2023-2024

---

*Documented by Harjas Singh - harjas@example.com*
```

This README includes:

1. **Comprehensive setup instructions** with environment variable configuration
2. **Clear usage examples** for all components (CLI, server, API)
3. **Troubleshooting section** for common issues
4. **Project structure** overview
5. **API documentation** links
6. **Contributing guidelines**
7. **Important disclaimers** about mental health support
8. **Developer information** with proper attribution

The `.env` setup is clearly explained, making it easy for anyone to get the project running quickly.Here's a comprehensive `README.md` file for your mental health chatbot project:

```markdown
# Mental Health Chatbot 🤖💚

A RAG-powered mental health assistant built with Groq API, Llama 3.3 70B, and ChromaDB. Provides empathetic, context-aware responses for mental health support while handling general queries.

## 🚀 Features

- **RAG Architecture**: Retrieval Augmented Generation for evidence-based mental health responses
- **Groq API Integration**: Ultra-fast inference with Llama 3.3 70B model
- **ChromaDB Vector Store**: Efficient knowledge retrieval and storage
- **Conversation Memory**: Maintains context across chat sessions
- **Dual Mode**: Mental health focus with general knowledge fallback
- **FastAPI Server**: Robust REST API with comprehensive logging
- **CLI Interface**: Interactive command-line chat interface

## 📁 Project Structure

```
mental-health-chatbot/
├── rag_layer.py          # RAG system with ChromaDB integration
├── agent.py             # CLI chat interface with memory
├── server.py            # FastAPI REST server
├── dataset.py           # Synthetic data generation (SDV)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd mental-health-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit the `.env` file and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🔑 Getting Groq API Key

1. Go to [Groq Console](https://console.groq.com/)
2. Sign up/login to your account
3. Navigate to API Keys section
4. Generate a new API key
5. Copy the key to your `.env` file

## 🚀 Usage

### Option 1: CLI Chat Interface
```bash
python agent.py
```

**Available Commands:**
- `/help` - Show help menu
- `/new` - Start new chat session
- `/history` - View conversation history
- `/save` - Save conversation to file
- `/exit` - Exit the program

### Option 2: FastAPI Server
```bash
python server.py
```

**API Endpoints:**
- `GET /` - Server information
- `GET /health` - Health check
- `POST /chat` - Send message to chatbot
- `GET /sessions` - List active sessions
- `GET /docs` - Interactive API documentation

### Option 3: Direct API Usage
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "I need help with anxiety"}'
```

## 🧠 How It Works

1. **Query Classification**: Determines if query is mental health-related
2. **RAG Retrieval**: For mental health queries, searches ChromaDB knowledge base
3. **Contextual Response**: Generates responses using retrieved context + Llama 3.3
4. **General Fallback**: For non-mental health queries, uses direct API calls
5. **Session Management**: Maintains conversation memory across interactions

## 📊 Knowledge Base

The system includes pre-loaded mental health knowledge about:
- Depression and anxiety disorders
- Cognitive Behavioral Therapy (CBT)
- Crisis resources and hotlines
- Mindfulness and coping techniques
- Self-care strategies

## 🔧 Configuration

### Environment Variables
```env
GROQ_API_KEY=your_groq_api_key_here
CHROMA_DB_PATH=./chroma_mentalhealth_db
LOG_LEVEL=INFO
```

### Custom Knowledge Base
Add your own mental health documents by modifying the sample data in:
- `rag_layer.py` - `initialize_mental_health_knowledge_base()`
- `server.py` - startup event

## 📋 API Documentation

Once the server is running, access interactive documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🐛 Troubleshooting

### Common Issues

1. **GROQ_API_KEY not set**
   ```bash
   export GROQ_API_KEY=your_actual_key_here
   ```

2. **Port already in use**
   ```bash
   # Change port in server.py or use:
   python server.py --port 8080
   ```

3. **ChromaDB errors**
   ```bash
   # Try resetting the database
   rm -rf chroma_mentalhealth_db/
   ```

### Logs
- Application logs: `mental_health_server.log`
- Access logs: Console output when running server

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Style
```bash
# Format code
black *.py

# Check linting
flake8 *.py
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This is a capstone project developed for educational purposes. It is not a substitute for professional mental health care. For serious mental health concerns, please contact licensed professionals or emergency services.

**Crisis Resources:**
- National Suicide Prevention Lifeline: 988 or 1-800-273-8255
- Crisis Text Line: Text "HOME" to 741741
- Emergency Services: 911

## 👥 Developers

**Group 33 - B.Tech CSE Cloud Computing & Automation**
- Harjas Partap Singh
- Divyam Sharma
- Sonia
- Manini
- Jiyaa

**Capstone Project** - Academic Year 2023-2024

---

*Documented by Harjas Singh - harjas42@icloud.com*
```

This README includes:

1. **Comprehensive setup instructions** with environment variable configuration
2. **Clear usage examples** for all components (CLI, server, API)
3. **Troubleshooting section** for common issues
4. **Project structure** overview
5. **API documentation** links
6. **Contributing guidelines**
7. **Important disclaimers** about mental health support
8. **Developer information** with proper attribution

The `.env` setup is clearly explained, making it easy for anyone to get the project running quickly.