# Local Vector Database Approach

## Overview

This approach creates a **persistent local vector database** from your DBT knowledge graph, eliminating the need for LM Studio embeddings and enabling **fast startup times**.

## How It Works

```
Data Processing (One-Time):
nodes.json + edges.json → Documents → Local Embeddings → Vector Index (saved locally)

Runtime (Fast):
User Question → Search Local Index → LM Studio LLM → Response
```

## Benefits

1. **One-Time Setup** - Generate embeddings once, use forever
2. **Fast Startup** - Load pre-built index in seconds
3. **Pure LM Studio** - Only LLM calls, no embedding API needed  
4. **Offline Ready** - Everything runs locally
5. **Better Performance** - Optimized vector search

## Setup Process

### Option 1: Pre-build Index (Recommended)

```bash
# 1. Install sentence-transformers
pip install sentence-transformers

# 2. Build the vector index (one-time, ~5-10 minutes)
python build_index.py

# 3. Run the backend (fast startup!)
python main.py
```

### Option 2: Build on First Run

```bash
# Just run the backend - it will build the index automatically
python main.py
```

## What Gets Created

- **Vector Index**: `./storage/index_store.json` - Searchable embeddings
- **Document Store**: `./storage/docstore.json` - Original documents  
- **Graph Store**: `./storage/graph_store.json` - Document relationships

## Performance Expectations

- **Index Building**: 5-10 minutes (one-time)
- **Startup Time**: 2-5 seconds (with pre-built index)
- **Query Response**: 1-3 seconds per question
- **Memory Usage**: ~500MB for embeddings

## Technical Details

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Size**: 90MB download
- **Speed**: ~1000 documents/minute
- **Quality**: Good for semantic search

### Vector Storage
- **Format**: LlamaIndex VectorStoreIndex
- **Search**: Cosine similarity 
- **Retrieval**: Top-K most relevant documents
- **Persistence**: JSON files for fast loading

## Data Flow

1. **Load DBT Data**: Parse `nodes.json` and `edges.json`
2. **Create Documents**: Convert to text documents with metadata
3. **Generate Embeddings**: Use SentenceTransformers locally
4. **Build Index**: Create searchable vector database
5. **Save Locally**: Persist for future use
6. **Runtime Search**: Query → Retrieve → LLM → Response

## Advantages vs LM Studio Embeddings

| Aspect | Local Vector DB | LM Studio Embeddings |
|--------|----------------|---------------------|
| **Setup Time** | 10 mins once | Every startup |
| **Startup Speed** | 2-5 seconds | 5-10 minutes |
| **Dependencies** | SentenceTransformers | LM Studio embedding model |
| **Reliability** | Always works | Depends on LM Studio |
| **Performance** | Optimized | Network calls |

## Troubleshooting

### If Index Building Fails
- Check that `nodes.json` and `edges.json` exist
- Ensure sufficient memory (2GB+ recommended)
- Try smaller batch size in `build_index.py`

### If Startup is Slow
- Pre-build the index with `python build_index.py`
- Check that `./storage/` contains index files

### If Search Quality is Poor
- Try different embedding model in `LocalEmbedding`
- Adjust `SIMILARITY_TOP_K` in config
- Review document chunking strategy
