# AI Role-Playing Chat Application

If you've played and are interested in cat box-like role-playing software, you'll likely be interested in this application. My original intention for creating this app was more out of interest and technical exercise, as well as due to the recent phenomenon of Cat Box requiring ads to chat and charging fees.

The recent update of Cat Box's diary and social media functions (paid) has also been reproduced with good results. Have you ever been frustrated by the memory loss issue common in companion apps? This application addresses that problem with optimized design for a more user-friendly experience.

This is a Flask-based AI role-playing agent developed using the langgraph agent architecture. It allows users to create and interact with AI characters through dialogue. The application integrates multiple large language model APIs, optimizes the character memory system, and includes social features such as Moments (social feed) and diaries.

## Key Features

### 1. Chat System
- During conversations, the agent can not only generate text replies but also determine whether a photo should be shared to enhance the visual experience.
- Image generation: The agent evaluates if an image needs to be shared during the chat. I use a prompt architecture combining few-shot learning and Chain-of-Thought (COT) to guide the LLM in converting chat content into professional image generation prompts, improving image quality.

![Project Image](聊天图片.png)

### 2. Social Features
- AI character Moments (social feed)
- AI character diary system
- The generation of Moments and diaries is managed using different threads in langgraph, allowing parallel generation without affecting chat performance. (Alternatively, a multi-agent architecture can be used for complex tasks, but since diary and Moments generation is relatively simple, merging them into one agent avoids the complexity of state transfer.)
- I use the `gemeni_pro` model and a COT-style prompt architecture to deeply analyze short-term and long-term memories, better understand key events and character personalities, and capture chat habits for writing Moments and diaries.

![Project Image](朋友圈1.png)



![Project Image](朋友圈2.png)



![Project Image](日记图片.png)

### 3. Memory System
- The memory module is divided into two parts: 1. Short-term memory 2. Dormant long-term memory.
- Short-term conversation memory: The threshold is set to 400 dialogue entries. Once exceeded, the memory is gradually cleared. Every 100 messages form a memory block, and the LLM generates several tags for each block, which are stored in the memory database.
- Long-term memory: Like human memory, when asked about past events, we recall the scene. Similarly, the agent first checks if the user's question relates to short-term memory. If not, it retrieves relevant long-term memories by matching tags, effectively preventing memory loss.
- Memory fusion mechanism: The memory module now primarily adopts a long-short term memory fusion approach. Through hook nodes in the langgraph workflow, short-term memory is automatically summarized and then stored in long-term memory.
- RAG integration: Long-term memory integrates RAG (Retrieval-Augmented Generation) using ChromaDB vector store and HuggingFace embeddings for memory retrieval, enabling more accurate and contextually relevant memory recall.
- Concurrent search: The project implements concurrent search for both long-term and short-term memory, improving retrieval efficiency.

### 4. AI Model Integration
- Support for multiple text generation LLM APIs:
  - Google Gemini
  - Qwen
  - Moonshot-Kimi
- Image generation:
  - Google Gemini (free, but no concurrency; suitable for personal use only)

## Tech Stack

### Backend
- Flask: Web framework
- SQLAlchemy: ORM and database operations
- Flask-Bcrypt: Password encryption
- JWT: User authentication
- Langchain/langgraph: LLM chains and agents
- SQLite: Data storage
- ChromaDB: Vector database for RAG implementation
- HuggingFace Embeddings: For vector embeddings

### Frontend
- Native JavaScript
- HTML5
- CSS3
- Server-Sent Events (SSE)

## Directory Structure

```
├── api_key.py              # API key configuration
├── app.py                  # Main Flask application
├── base.py                 # Base configuration and LLM settings
├── chat_data.db            # Chat database (stores all chat logs, Moments, and diaries for characters)
├── generate_content.py     # Content generation for chats, images, Moments, and diaries
├── get_memory.py           # Memory system - SQLite database operations for character profiles and chat memories
├── get_character_full_data.py # Database operations for chat history, social posts, and diary entries
├── memory_data.db          # Memory database for long-term memories
├── main_agent.py           # Langgraph agent workflow definition
├── memory.py               # Memory management with RAG integration using ChromaDB
├── state.py                # State definitions for the langgraph agent
├── requirements.txt        # Python dependencies
├── static/                 # Static files
│   ├── index.html
│   ├── script.js
│   └── style.css
│     └── assets
│            └── default_avatar.png (default character avatar, customizable)
│            └── user_hand_portrait.jpg  (default user avatar)
│
├── uploads/                # User-uploaded character avatars
└── talk_picture/           # AI-generated images
```

## Running Instructions

1. Install dependencies (install any missing ones manually):
```bash
pip install -r requirements.txt
```

2. Configure API keys: Edit the `api_key.py` file and fill in the required API keys. (If you don't have a certain key, override it in `base.py` to avoid errors. Note that performance may vary.)

3. Run the application:
```bash
python app.py
```

4. Access the application:
Open your browser and go to `http://localhost:5000`

## Notes
- Required database files will be created automatically on first run.
- Ensure that the `uploads` and `talk_picture` directories have write permissions.
- Valid API keys for AI services are required to use all features.

## Areas for Improvement:
- Poor concurrency: Since most people are unwilling to pay for Google models (too expensive), and although Google models are free, they have concurrency and speed limits, I did not implement asynchronous development. The entire project is synchronous. My original intention was to run it locally.
- Text generation takes about 3-5 seconds (if an image is needed, it increases to ~8 seconds). Unlike simple chatbots, this project involves many additional steps (e.g., long-term memory retrieval, image generation), which adds processing time but enhances quality and experience.
- No real-time text streaming: Since the reply text is generated within langgraph nodes, it's not straightforward to stream the output (would require callbacks or external message queues, which is complex - so I skipped it for simplicity!).

## Summary

This project is ideal for those learning langchain or langgraph for agent development, or for exploring prompt engineering (few-shot, COT, TOT) and workflow design. It also has strong potential for further development. For example, to enrich character personalities, you could integrate APIs for daily life information (e.g., trending topics, weather, search) and transform internal chains into react_agents. Additionally, the diary and Moments generation can be separated from the main agent and developed as independent agents using multi-agent architectures like swarm or supervisor, with research into faster and better inter-agent communication.