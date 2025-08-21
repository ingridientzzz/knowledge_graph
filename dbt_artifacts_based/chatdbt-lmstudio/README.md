# ChatDBT with Local LLM

A conversational AI application that allows users to chat with their dbt knowledge graph using LM Studio for local LLM inference.

## Overview

This project processes dbt artifacts (manifest.json, catalog.json, graph_summary.json) and creates an intelligent assistant that can answer questions about your data models, transformations, dependencies, and relationships using a local LM Studio server.

## Features

- **Local LM Studio Integration** - Uses LM Studio's OpenAI-compatible API
- **Knowledge Graph Processing** - Analyzes dbt models, sources, and relationships from artifacts
- **Conversational Interface** - Modern React-based chat UI
- **Source Attribution** - Shows which nodes/models informed each response
- **Real-time Updates** - Refresh index when data changes
- **Rich Context Understanding** - Knows about model relationships, dependencies, and metadata

## Architecture

- **Backend**: FastAPI with LlamaIndex for document processing and RAG
- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **LLM**: LM Studio server (OpenAI-compatible API)
- **Data**: Processes manifest.json, catalog.json, and graph_summary.json from dbt artifacts

## Setup

### Prerequisites

1. **Install LM Studio**
   - Download from [lmstudio.ai](https://lmstudio.ai/)
   - Load a model (e.g., Code Llama, Mistral, etc.)
   - Start the local server (default: http://localhost:1234)

2. **Prepare your data**
   - Ensure you have `manifest.json`, `catalog.json`, and `graph_summary.json`
   - These should be from your dbt project artifacts (typically in the `code_docs/` folder)

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   Create `.env` file in `/backend`:
   ```ini
   DBT_ARTIFACTS_PATH=../../code_docs
   LM_STUDIO_BASE_URL=http://localhost:1234/v1
   LM_STUDIO_API_KEY=lm-studio
   HOST=0.0.0.0
   PORT=8000
   INDEX_STORAGE_PATH=./storage
   USE_PROPERTY_GRAPH=true
   ```

3. **Run the backend server:**
   ```bash
   cd backend
   python main.py
   ```

### Frontend Setup

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Run the frontend:**
   ```bash
   npm run dev
   ```

3. **Open the application:**
   Open [http://localhost:3000](http://localhost:3000)

## Usage

Once both servers are running, you can ask questions like:

- "What models are available in the data warehouse?"
- "Show me the dependencies for the customer model"
- "What columns are in the revenue table?"
- "How are the engagement models related?"
- "What tests are applied to the user data?"

## Configuration

All configuration is handled through environment variables:

- `DBT_ARTIFACTS_PATH`: Path to the folder containing manifest.json, catalog.json, and graph_summary.json
- `LM_STUDIO_BASE_URL`: LM Studio server URL (default: http://localhost:1234/v1)
- `LM_STUDIO_API_KEY`: API key for LM Studio (default: lm-studio)
- `INDEX_STORAGE_PATH`: Where to store the vector index
- `USE_PROPERTY_GRAPH`: Whether to use PropertyGraph (true) or VectorStore (false) indexing (default: true)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

## API Endpoints

- `GET /`: Health check
- `GET /health`: Detailed health status
- `POST /chat`: Chat with your dbt knowledge graph
- `POST /refresh-index`: Refresh the index with latest data files

## Troubleshooting

1. **LM Studio connection issues:**
   - Ensure LM Studio is running with server enabled
   - Check the base URL in your configuration
   - Verify a model is loaded in LM Studio

2. **No documents found:**
   - Verify `DBT_ARTIFACTS_PATH` points to correct folder
   - Ensure manifest.json, catalog.json, and graph_summary.json exist and are readable

3. **Performance issues:**
   - Use smaller models in LM Studio for faster responses
   - Adjust chunk size in the configuration
   - Consider reducing the dataset size for testing
