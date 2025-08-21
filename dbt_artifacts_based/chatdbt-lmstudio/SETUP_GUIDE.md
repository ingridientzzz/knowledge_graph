# ChatDBT with Local LLM - Setup Guide

## Quick Start

This guide will help you set up ChatDBT with Local LLM to chat with your dbt knowledge graph.

### Prerequisites Checklist

- [ ] **LM Studio installed** from [lmstudio.ai](https://lmstudio.ai/)
- [ ] **Model loaded in LM Studio** (e.g., Code Llama, Mistral, Qwen, etc.)
- [ ] **LM Studio server running** on default port 1234
- [ ] **Data files available**: `manifest.json`, `catalog.json`, `graph_summary.json`

### Step 1: Install LM Studio

1. Download LM Studio from [lmstudio.ai](https://lmstudio.ai/)
2. Install and open LM Studio
3. Browse and download a suitable model:
   - **Recommended for coding**: `microsoft/CodeGPT-small-java-adaptedGPT2` or `codellama/CodeLlama-7b-Instruct-GGUF`
   - **Recommended for general use**: `mistralai/Mistral-7B-Instruct-v0.2-GGUF`
   - **For better reasoning**: `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`

### Step 2: Start LM Studio Server

1. In LM Studio, go to the **Local Server** tab
2. Select your loaded model
3. Click **Start Server**
4. Verify it's running on `http://localhost:1234` (default)

### Step 3: Configure Backend

1. Navigate to the backend directory:
   ```bash
   cd chatdbt-lmstudio/backend
   ```

2. Copy the environment template:
   ```bash
   cp env_example.txt .env
   ```

3. Edit `.env` file with your configuration:
   ```ini
   # LM Studio Configuration
   LM_STUDIO_BASE_URL=http://localhost:1234/v1
   LM_STUDIO_API_KEY=lm-studio
   LM_STUDIO_MODEL=local-model

   # Data Configuration - Points to the dbt artifacts directory (two levels up)
   DBT_ARTIFACTS_PATH=../../code_docs

   # Server Configuration
   HOST=0.0.0.0
   PORT=8000
   INDEX_STORAGE_PATH=./storage
   ```

   **✅ Note**: `DBT_ARTIFACTS_PATH` is set to `../../code_docs` since the project is located within the `dbt_artifacts_based/chatdbt-lmstudio` directory and uses dbt artifacts from the code_docs folder.

### Step 4: Start the Backend

Option 1 - Using the startup script:
```bash
./start_backend.sh
```

Option 2 - Manual setup:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Expected output:**
```
Loading manifest.json...
Loaded manifest with 1234 nodes
Loading graph_summary.json...
Loaded graph summary with 567 linked nodes
Loading catalog.json...
Loaded catalog with 890 cataloged nodes
Created 123 documents with property graph data
LlamaIndex initialized successfully with LM Studio.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 5: Start the Frontend

Option 1 - Using the startup script:
```bash
./start_frontend.sh
```

Option 2 - Manual setup:
```bash
cd frontend
npm install
npm run dev
```

**Expected output:**
```
- Ready on http://localhost:3000
```

### Step 6: Test the Application

1. Open your browser to [http://localhost:3000](http://localhost:3000)
2. You should see the ChatDBT interface
3. Check the connection status (green dot = connected)
4. Try asking a question like: "What models are available in the data warehouse?"

## Troubleshooting

### Backend Issues

**"Cannot connect to LM Studio"**
- Ensure LM Studio server is running on port 1234
- Check that a model is loaded and selected
- Verify the `LM_STUDIO_BASE_URL` in your `.env` file

**"No documents found"**
- Check that `DBT_ARTIFACTS_PATH` in `.env` points to the correct directory
- Verify that `manifest.json`, `catalog.json`, and `graph_summary.json` exist
- Check file permissions (files should be readable)

**"Module not found" errors**
- Ensure you're in the correct virtual environment
- Run `pip install -r requirements.txt` again
- Try updating pip: `pip install --upgrade pip`

### Frontend Issues

**"Cannot connect to backend"**
- Ensure the backend is running on port 8000
- Check console errors in browser developer tools
- Verify CORS settings if accessing from different domains

**Frontend build errors**
- Ensure Node.js version is 18+ 
- Delete `node_modules` and run `npm install` again
- Clear Next.js cache: `npm run build` then `npm run dev`

### Model Performance

**Slow responses**
- Try a smaller model in LM Studio
- Reduce `SIMILARITY_TOP_K` in `.env` (default: 10)
- Increase `LLM_CHUNK_SIZE` for better context (default: 4096)

**Poor quality responses**
- Try a different model (Code Llama works well for technical content)
- Adjust `TEMPERATURE` (lower = more focused, higher = more creative)
- Check if the model has enough context length for your data

## Model Recommendations

### For Technical/Code Analysis
- **CodeLlama-7B-Instruct**: Best for understanding SQL and code relationships
- **Qwen2.5-Coder-7B**: Excellent for technical documentation
- **DeepSeek-Coder-6.7B**: Good balance of speed and accuracy

### For General Data Questions
- **Mistral-7B-Instruct**: Good general-purpose model
- **Llama-2-7B-Chat**: Reliable for conversational queries
- **Phi-3-Mini**: Fast and efficient for basic questions

## Advanced Configuration

### Custom Chunking Strategy
```ini
# In .env file
LLM_CHUNK_SIZE=8192          # Larger chunks for more context
LLM_CHUNK_OVERLAP=400        # More overlap for better continuity
SIMILARITY_TOP_K=15          # More sources for comprehensive answers
```

### Performance Tuning
```ini
# For faster responses
LLM_CHUNK_SIZE=2048
SIMILARITY_TOP_K=5
TEMPERATURE=0.0

# For more comprehensive answers
LLM_CHUNK_SIZE=8192
SIMILARITY_TOP_K=20
TEMPERATURE=0.2
```

## Support

If you encounter issues:

1. Check the console logs in both backend and frontend
2. Verify LM Studio is properly configured and running
3. Ensure your data files are valid JSON and readable
4. Test with a simple question first: "What is this knowledge graph about?"

For additional help, check the LM Studio documentation and ensure your model is compatible with OpenAI API format.
