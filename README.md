# Korean RAG Chatbot

A Korean-optimized RAG (Retrieval-Augmented Generation) chatbot that connects to multiple data sources and answers questions using local LLM via Ollama.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Question                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                            │
│  ┌──────────┬──────────┬──────────┬──────────┐              │
│  │Document  │ Obsidian │   Jira   │  Notion  │              │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘              │
└───────┼──────────┼──────────┼──────────┼────────────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────┐
│              Korean Text Chunking & Processing               │
│         (Sentence boundary aware: 다., 요., 습니다.)         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Ollama Embeddings                         │
│                      (bge-m3 model)                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    ChromaDB Vector Store                     │
│                   (Cosine Similarity Search)                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Ollama LLM                              │
│                    (gemma3:1b model)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Korean Response                           │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Data Sources
- **Multi doc Upload**: Upload and query documents
- **Obsidian Vault**: Connect to local Obsidian markdown files
- **Jira**: Fetch and query Jira issues (Cloud & Server)
- **Notion**: Search and query Notion pages

### Korean Optimization
- Sentence-aware text chunking for Korean (다., 요., 습니다., etc.)
- Bilingual status keyword support (done/완료, progress/진행중)
- Korean UI interface

### Smart Features
- Auto-detection of status queries for Jira (done/완료/진행중/대기)
- Expandable issue/page lists in sidebar
- Source attribution in responses

## Requirements

- Python 3.10 (recommended)
- [Ollama](https://ollama.ai/) installed and running
- Required Ollama models:
  - `gemma3:1b` (LLM)
  - `bge-m3` (Embeddings)

## Installation

### 1. Clone and Setup Virtual Environment

```powershell
# Create virtual environment with Python 3.10
py -3.10 -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Or for Command Prompt
.\venv\Scripts\activate.bat
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install Ollama Models

```powershell
ollama pull gemma3:1b
ollama pull bge-m3
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```powershell
copy .env.example .env
```

Edit `.env` with your settings:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=gemma3:1b
OLLAMA_EMBEDDING_MODEL=bge-m3

# Obsidian Configuration
OBSIDIAN_VAULT_PATH=C:/Users/username/Documents/Obsidian Vault

# Jira Configuration
JIRA_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-api-token

# Notion Configuration
NOTION_API_TOKEN=secret_xxx

# RAG Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K_RESULTS=5
```

## Running the Application

### 1. Start Ollama

Make sure Ollama is running:

```powershell
ollama serve
```

### 2. Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Run Streamlit App

```powershell
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### PDF Upload
1. Select "PDF 업로드" from the sidebar
2. Upload a PDF file
3. Wait for indexing to complete
4. Ask questions in the chat

### Obsidian Vault
1. Select "Obsidian 볼트" from the sidebar
2. Enter your vault path (or use the default from .env)
3. Select a folder
4. Click "인덱싱 시작"
5. Ask questions about your notes

### Jira Integration
1. Select "Jira" from the sidebar
2. Credentials are auto-loaded from .env (or enter manually)
3. Click "연결 테스트" to verify connection
4. Select a project or enter custom JQL
5. Click "이슈 가져오기"
6. Ask questions like:
   - "완료된 작업 목록" (done tasks)
   - "진행중인 이슈" (in-progress issues)
   - "SEND-22에 대해 설명해줘"

### Notion Integration
1. Select "Notion" from the sidebar
2. Enter your Integration Token
3. Click "연결 테스트"
4. Search or fetch all pages
5. Click "페이지 가져오기"
6. Ask questions about your Notion content

**Note**: You must share your Notion pages with the integration for access.

## API Token Setup

### Jira API Token
1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Create API token
3. Use your email as username and the token as password

### Notion Integration Token
1. Go to https://www.notion.so/my-integrations
2. Create new integration
3. Copy the Internal Integration Token
4. Share your Notion pages with the integration

## Project Structure

```
chatbot_mvp/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
├── .env.example          # Example environment file
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── etl/
│   └── embedding_generation.py  # Ollama embedding wrapper
└── _party/               # Third-party integrations
    ├── __init__.py
    ├── obsidian.py       # Obsidian vault integration
    ├── jira_client.py    # Jira API client (v3)
    └── notion_client.py  # Notion API client
```

## Troubleshooting

### "Module not found" errors
```powershell
pip install -r requirements.txt
```

### Ollama connection error
Make sure Ollama is running:
```powershell
ollama serve
```

### Jira API 410 error
The app uses Jira API v3. If you see deprecation errors, ensure your Jira instance supports API v3.

### Memory issues with large models
Switch to a smaller model in `.env`:
```env
OLLAMA_LLM_MODEL=gemma3:1b
```

### ChromaDB collection name errors
Collection names are automatically sanitized. If issues persist, try restarting the app.

## License

MIT License
