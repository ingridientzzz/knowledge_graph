import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pydantic import BaseModel
from typing import List, Optional

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from sentence_transformers import SentenceTransformer
from typing import List, Any, Optional
import pickle
import numpy as np
import openai
import threading
import time
import uuid
import aiohttp
import json as json_module
import asyncio

from config import Config
from dbt_artifacts_loader import DBTArtifactsLoader
import json
from pathlib import Path
from query_router import route_query, QueryMethod
from graph_query_helper import GraphQueryHelper

class LocalEmbedding(BaseEmbedding):
    """Local embedding class using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embed_batch_size: int = 32):
        super().__init__(embed_batch_size=embed_batch_size)
        print(f"Loading local embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        print("Local embedding model loaded successfully")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query."""
        embedding = self._model.encode([query])
        return embedding[0].tolist()
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of get_query_embedding."""
        return self._get_query_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self._get_query_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self._model.encode(texts, show_progress_bar=True, batch_size=32)
        return [emb.tolist() for emb in embeddings]


class LMStudioLLM(CustomLLM):
    """Custom LLM for LM Studio that bypasses OpenAI model validation."""
    
    model_name: str = "local-model"
    temperature: float = 0.1
    max_tokens: int = 512
    api_base: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    timeout: float = 120.0
    
    def __init__(
        self,
        model: str = "local-model",
        api_base: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: float = 120.0,
        **kwargs
    ):
        super().__init__(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            **kwargs
        )
        
        # Initialize OpenAI client for LM Studio
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout
        )
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=4096,  # Adjust based on your model
            num_output=self.max_tokens,
            model_name=self.model_name,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete a prompt using LM Studio."""
        try:
            # Only pass basic parameters to avoid LM Studio compatibility issues
            # Filter out any parameters that might cause issues
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
                # Deliberately not passing **kwargs to avoid compatibility issues
            )
            
            return CompletionResponse(
                text=response.choices[0].message.content,
                raw=response,
            )
        except Exception as e:
            raise RuntimeError(f"LM Studio completion failed: {e}")
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream complete a prompt using LM Studio."""
        try:
            # Only pass basic parameters to avoid LM Studio compatibility issues
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
                # Deliberately not passing **kwargs to avoid compatibility issues
            )
            
            def gen() -> CompletionResponseGen:
                text = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        text += chunk.choices[0].delta.content
                        yield CompletionResponse(
                            text=text,
                            delta=chunk.choices[0].delta.content,
                            raw=chunk,
                        )
            
            return gen()
        except Exception as e:
            raise RuntimeError(f"LM Studio streaming completion failed: {e}")


app = FastAPI(title="ChatDBT with Local LLM")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize configuration
config = Config()

# Global variables
chat_engine = None
index = None
current_active_model = None
graph_helper = None

# Active request tracking for cancellation
active_requests = {}
request_lock = threading.Lock()

# Track active aiohttp sessions for cancellation
active_sessions = {}

async def stream_from_lm_studio_direct(messages, model, request_id, config):
    """Make direct HTTP request to LM Studio that can be cancelled."""
    print(f"🔍 DEBUG: config type: {type(config)}")
    print(f"🔍 DEBUG: config hasattr 'get': {hasattr(config, 'get')}")
    print(f"🔍 DEBUG: config content: {config}")
    
    # Handle both dict and object config
    base_url = config.get('LM_STUDIO_BASE_URL') if hasattr(config, 'get') else config.LM_STUDIO_BASE_URL
    api_key = config.get('LM_STUDIO_API_KEY') if hasattr(config, 'get') else config.LM_STUDIO_API_KEY
    
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Handle temperature and max_tokens config access
    temperature = config.get('TEMPERATURE', 0.1) if hasattr(config, 'get') else config.TEMPERATURE
    max_tokens = config.get('MAX_TOKENS', 2048) if hasattr(config, 'get') else config.MAX_TOKENS
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    print(f"🌐 Making direct HTTP request to LM Studio: {url}")
    print(f"📋 Payload: {payload}")
    
    session = None
    try:
        # Create timeout for the session
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes max
        session = aiohttp.ClientSession(timeout=timeout)
        
        # Store session for cancellation
        with request_lock:
            active_sessions[request_id] = session
            
        async with session.post(url, json=payload, headers=headers) as response:
            print(f"📡 LM Studio response status: {response.status}")
            
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"LM Studio returned {response.status}: {error_text}")
            
            # Read response line by line for Server-Sent Events (SSE)
            buffer = b''
            async for chunk in response.content.iter_chunked(1024):
                # Check for cancellation on each chunk
                with request_lock:
                    if request_id in active_requests and active_requests[request_id]['cancel_requested']:
                        print(f"🛑 HTTP CANCELLATION: Stopping LM Studio request for {request_id}")
                        raise asyncio.CancelledError("Request cancelled by user")
                
                buffer += chunk
                lines = buffer.split(b'\n')
                buffer = lines[-1]  # Keep incomplete line in buffer
                
                for line in lines[:-1]:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            return
                        try:
                            data = json_module.loads(data_str)
                            
                            # Check for errors first
                            if 'error' in data:
                                error_msg = data['error'].get('message', 'Unknown error from LM Studio')
                                print(f"❌ LM Studio error: {error_msg}")
                                raise Exception(f"LM Studio error: {error_msg}")
                            
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json_module.JSONDecodeError:
                            continue  # Skip invalid JSON lines
                        
    except asyncio.CancelledError:
        print(f"🛑 HTTP request to LM Studio cancelled for request {request_id}")
        raise
    except Exception as e:
        print(f"❌ Error in direct LM Studio request: {e}")
        raise
    finally:
        # Cleanup session
        if session:
            await session.close()
        with request_lock:
            if request_id in active_sessions:
                del active_sessions[request_id]

class ChatMessage(BaseModel):
    message: str
    context_size: Optional[int] = 20000

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None

def initialize_llama_index():
    """Initialize LlamaIndex with LM Studio."""
    global chat_engine, index

    try:
        print("Initializing LlamaIndex with LM Studio...")
        
        # Configure LM Studio LLM (Custom implementation)
        llm = LMStudioLLM(
            model="openai/gpt-oss-20b",  # Use the actual loaded model
            api_key=config.LM_STUDIO_API_KEY,
            api_base=config.LM_STUDIO_BASE_URL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            timeout=120.0
        )

        # Configure local embeddings (using SentenceTransformers)
        embed_model = LocalEmbedding(model_name="all-MiniLM-L6-v2")

        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = config.LLM_CHUNK_SIZE
        Settings.chunk_overlap = config.LLM_CHUNK_OVERLAP

        # Try to load existing index
        storage_path = Path(config.INDEX_STORAGE_PATH)

        if storage_path.exists() and (storage_path / "index_store.json").exists():
            print(f"Loading existing index from {storage_path}...")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))
                index = load_index_from_storage(storage_context)
                print("Successfully loaded existing index")
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                print("Creating new index...")
                index = None

        if index is None:
            print("Creating new index from dbt artifacts...")
            
            if config.USE_PROPERTY_GRAPH:
                print("Using Hybrid VectorStore approach with PropertyGraph data...")
                # Load dbt artifacts for property graph data
                artifacts_loader = DBTArtifactsLoader(config.DBT_ARTIFACTS_PATH)
                documents, entity_nodes, relations = artifacts_loader.load_property_graph_data()

                if not documents:
                    raise Exception("No documents found in dbt artifacts. Please check your DBT_ARTIFACTS_PATH.")

                print(f"Loaded {len(documents)} documents with {len(entity_nodes)} entities and {len(relations)} relations")
                print("Creating VectorStore index from PropertyGraph documents (true hybrid approach)...")
                print("This allows efficient vector retrieval with rich graph-structured data.")

                # Create VectorStore index with PropertyGraph-derived documents
                # This is the true hybrid approach: vector retrieval + graph-structured content
                index = VectorStoreIndex.from_documents(
                    documents,
                    show_progress=True
                )
                
                print(f"Hybrid VectorStore index created with {len(documents)} documents")
                print(f"Includes {len(entity_nodes)} entities and {len(relations)} relations from PropertyGraph data")
                
            else:
                print("Using VectorStoreIndex approach...")
                # Load dbt artifacts for document creation only
                artifacts_loader = DBTArtifactsLoader(config.DBT_ARTIFACTS_PATH)
                documents = artifacts_loader.load_documents()

                if not documents:
                    raise Exception("No documents found in dbt artifacts. Please check your DBT_ARTIFACTS_PATH.")

                # Validate that documents are LlamaIndex Document objects
                if not all(isinstance(doc, Document) for doc in documents):
                    raise TypeError("DBTArtifactsLoader must return a list of llama_index.core.schema.Document objects.")

                print(f"Loaded {len(documents)} documents from dbt artifacts")

                # Create vector index
                index = VectorStoreIndex.from_documents(documents)

            # Persist index
            storage_path.mkdir(parents=True, exist_ok=True)
            index.storage_context.persist(persist_dir=str(storage_path))
            print(f"Index persisted to {storage_path}")

        # Create chat engine with enhanced capabilities
        if config.USE_PROPERTY_GRAPH:
            print("Creating PropertyGraph-aware chat engine...")
            # Use PropertyGraph index's built-in retriever that can access graph data
            retriever = index.as_retriever(
                similarity_top_k=config.SIMILARITY_TOP_K,
                # PropertyGraphIndex retriever includes graph relationship access
            )
            
            # Generate dynamic system message based on actual loaded data
            entity_count = len(entity_nodes) if 'entity_nodes' in locals() else "thousands of"
            relation_count = len(relations) if 'relations' in locals() else "thousands of"
            
            chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever,
                system_message=(
                    "You are a dbt project analyst with access to a comprehensive PropertyGraph knowledge base. "
                    "You can query both documents AND graph relationships directly. "
                    "For statistical queries like 'What packages have the most models?', provide exact counts from the graph data. "
                    "For lineage queries, trace actual dependencies through the graph relationships. "
                    "When asked about packages, models, tests, or sources, give specific names and counts. "
                    f"You have access to {entity_count} entities and {relation_count} relationships across multiple dbt packages. "
                    "The graph contains various packages with models, tests, sources, and seeds. "
                    "Always provide precise, data-driven answers with exact statistics when available."
                )
            )
        else:
            print("Creating standard chat engine...")
            chat_engine = CondensePlusContextChatEngine.from_defaults(
                index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K),
                system_message=(
                    "You are a helpful assistant specialized in analyzing dbt (data build tool) projects and knowledge graphs. "
                    "You have access to comprehensive information about data models, their relationships, dependencies, "
                    "SQL transformations, tests, sources, and metadata. "
                    "When answering questions, provide specific information about the models, their relationships, "
                    "and how they transform data. Reference specific model names, dependencies, and SQL logic when relevant. "
                    "If asked about data lineage or dependencies, explain the full chain of relationships. "
                    "For questions about data quality, mention relevant tests and validations. "
                    "Always be specific and reference the actual model/table/column names from the knowledge graph."
                )
            )

        print("LlamaIndex initialized successfully with LM Studio.")
        
        # Initialize the current active model
        global current_active_model
        current_active_model = config.LM_STUDIO_MODEL
        print(f"🎯 Initial active model set to: {current_active_model}")

    except Exception as e:
        print(f"Error initializing LlamaIndex: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    initialize_llama_index()

@app.get("/")
async def root():
    return {"message": "ChatDBT with Local LLM API is running."}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test LM Studio connection
        llm_test = LMStudioLLM(
            model="openai/gpt-oss-20b",
            api_key=config.LM_STUDIO_API_KEY,
            api_base=config.LM_STUDIO_BASE_URL,
            timeout=5.0,
            max_tokens=5
        )
        _ = llm_test.complete("Hello")

        return {
            "status": "healthy",
            "lm_studio_model": config.LM_STUDIO_MODEL,
            "lm_studio_host": config.LM_STUDIO_BASE_URL,
            "dbt_artifacts_path": config.DBT_ARTIFACTS_PATH,
            "index_loaded": index is not None,
            "chat_engine_ready": chat_engine is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "lm_studio_host": config.LM_STUDIO_BASE_URL
        }

@app.post("/cancel/{request_id}")
async def cancel_request(request_id: str):
    """Cancel an active request by ID - Enhanced cancellation with immediate effect."""
    with request_lock:
        if request_id in active_requests:
            active_requests[request_id]['cancel_requested'] = True
            active_requests[request_id]['status'] = 'cancelling'
            print(f"🛑 STOP BUTTON: Cancellation requested for request {request_id}")
            
            # Close any active HTTP session to LM Studio
            if request_id in active_sessions:
                session = active_sessions[request_id]
                if not session.closed:
                    print(f"🛑 STOP BUTTON: Forcibly closing LM Studio HTTP session for request {request_id}")
                    await session.close()
                del active_sessions[request_id]
            
            return {
                "status": "cancelled", 
                "message": f"Request {request_id} cancelled - LM Studio connection closed",
                "request_id": request_id
            }
        else:
            print(f"❌ STOP BUTTON: Request {request_id} not found in active requests")
            return {"status": "not_found", "error": f"Request {request_id} not found"}

@app.post("/chat")
async def chat(message: ChatMessage, request: Request):
    """Chat endpoint with streaming support and cancellation."""
    global graph_helper, current_active_model, config
    
    if not chat_engine:
        raise HTTPException(status_code=500, detail="Chat engine not initialized. Please check server logs.")
    
    if not index:
        raise HTTPException(status_code=500, detail="Vector index not loaded. Please check server logs.")

    # Generate unique request ID for cancellation tracking
    request_id = str(uuid.uuid4())
    
    print(f"Chat engine type: {type(chat_engine)}")
    print(f"Index type: {type(index)}")
    print(f"Index loaded: {index is not None}")

    try:
        from fastapi.responses import StreamingResponse
        import json
        
        async def generate_response():
            """Async generator function for streaming response with cancellation support."""
            
            # Register this request for cancellation tracking
            with request_lock:
                active_requests[request_id] = {
                    'status': 'starting',
                    'cancel_requested': False,
                    'start_time': time.time()
                }
            
            try:
                print(f"Starting streaming response for: {message.message} (ID: {request_id})")
                
                # 🧠 SMART QUERY ANALYSIS (LLM will decide how to use this info)
                routing_info = route_query(message.message)
                intent = routing_info['intent']
                suggested_method = routing_info['method']
                routing_config = routing_info['config']
                
                print(f"🎯 Query Intent: {intent}")
                print(f"💡 Suggested Method: {suggested_method}")
                print(f"🔍 Config: {routing_config}")
                
                # 🤖 ENHANCED CONTEXT PREPARATION - Give LLM access to graph query results if relevant
                enhanced_context_parts = []
                graph_query_results = {}
                
                # If this looks like a dependency or graph query, provide graph query results as context
                if intent in ['graph_dependency', 'graph_search', 'graph_commonality']:
                    print("🔍 Pre-fetching graph query results for LLM context...")
                    try:
                        local_graph_helper = GraphQueryHelper(config.INDEX_STORAGE_PATH)
                        
                        if intent == 'graph_dependency':
                            model_name = routing_config.get('model_name', '')
                            is_upstream = routing_config.get('is_upstream_query', False)
                            
                            if model_name:
                                print(f"📊 Fetching dependency data for: {model_name}")
                                if is_upstream:
                                    graph_query_results['upstream'] = local_graph_helper.find_upstream(model_name)
                                    enhanced_context_parts.append(f"GRAPH QUERY RESULT - Upstream dependencies for {model_name}:")
                                    enhanced_context_parts.append(json.dumps(graph_query_results['upstream'], indent=2))
                                else:
                                    graph_query_results['downstream'] = local_graph_helper.find_dependencies(model_name)
                                    enhanced_context_parts.append(f"GRAPH QUERY RESULT - Downstream dependencies for {model_name}:")
                                    enhanced_context_parts.append(json.dumps(graph_query_results['downstream'], indent=2))
                        
                        elif intent == 'graph_search':
                            search_term = routing_config.get('search_term', '')
                            if search_term:
                                print(f"🔍 Fetching search results for: {search_term}")
                                models = local_graph_helper.search_models(search_term)
                                graph_query_results['search'] = {'term': search_term, 'models': models}
                                enhanced_context_parts.append(f"GRAPH QUERY RESULT - Models containing '{search_term}':")
                                enhanced_context_parts.append(f"Found {len(models)} models: {models[:20]}")
                                if len(models) > 20:
                                    enhanced_context_parts.append(f"... and {len(models)-20} more models")
                        
                        elif intent == 'graph_commonality':
                            entities = routing_config.get('entities', [])
                            if len(entities) >= 2:
                                print(f"🔗 Fetching commonality analysis for: {entities}")
                                commonalities = local_graph_helper.find_commonalities(entities)
                                graph_query_results['commonalities'] = commonalities
                                enhanced_context_parts.append(f"GRAPH QUERY RESULT - Commonalities between {entities}:")
                                enhanced_context_parts.append(json.dumps(commonalities, indent=2))
                            else:
                                enhanced_context_parts.append(f"Note: Could not extract enough entities for commonality analysis from query. Found: {entities}")
                    
                    except Exception as e:
                        print(f"⚠️ Graph query failed, LLM will use vector search: {e}")
                        enhanced_context_parts.append(f"Note: Graph query failed ({str(e)}), using vector search fallback.")
                
                print("Checking chat engine availability...")
                
                # Check if request is already cancelled
                if await request.is_disconnected():
                    print("Client disconnected before processing started")
                    with request_lock:
                        active_requests[request_id]['status'] = 'cancelled_early'
                    return
                
                # Check internal cancellation flag
                with request_lock:
                    if active_requests[request_id]['cancel_requested']:
                        print("Request cancelled via stop button before processing")
                        active_requests[request_id]['status'] = 'cancelled_early'
                        return
                
                # Test chat engine first
                if not hasattr(chat_engine, 'stream_chat'):
                    raise Exception("Chat engine does not have stream_chat method")
                
                print("Chat engine has stream_chat method")
                
                # 🚀 HYBRID APPROACH - Skip chat engine, use direct retrieval
                print(f"🔄 Using HYBRID APPROACH with optimized retrieval for intent: {intent}")
                print(f"Creating optimized retriever with top_k={routing_config['similarity_top_k']}")
                
                # Special handling for package list queries
                if intent == 'package_list':
                    print("🎯 Package list query detected - using enhanced search")
                    # For package queries, we want to ensure we get ALL package batch documents
                    optimized_retriever = index.as_retriever(
                        similarity_top_k=30,  # High number to ensure all package batches are retrieved
                        node_postprocessors=[],  # No filtering to ensure we get all relevant docs
                    )
                elif intent == 'model_search' and any(pkg in message.message.lower() for pkg in ['package', 'pkg']):
                    print("🎯 Package-specific model search detected - using enhanced search")
                    # For package model queries, prioritize package model directories
                    optimized_retriever = index.as_retriever(
                        similarity_top_k=25,  # High number to get package directories + individual models
                        node_postprocessors=[],  # No filtering to ensure we get package directories
                    )
                else:
                    optimized_retriever = index.as_retriever(
                        similarity_top_k=routing_config['similarity_top_k']
                    )
                
                # Skip temporary chat engine - go directly to hybrid approach
                print("⚡ Skipping chat engine, going directly to hybrid retrieval...")
                
                # Mark as processing
                with request_lock:
                    active_requests[request_id]['status'] = 'processing'
                
                # 🔄 HYBRID APPROACH: Get context from LlamaIndex, stream directly from LM Studio
                try:
                    # First, retrieve relevant context using the optimized retriever
                    print("🔍 Retrieving context using optimized retriever...")
                    retrieved_nodes = optimized_retriever.retrieve(message.message)
                    print(f"📚 Retrieved {len(retrieved_nodes)} context nodes")
                    
                    # Build context string from retrieved nodes
                    context_parts = []
                    for i, node in enumerate(retrieved_nodes):
                        metadata = node.metadata
                        content = node.get_content()
                        
                        context_parts.append(f"Document {i+1}:")
                        context_parts.append(f"Source: {metadata.get('name', 'Unknown')}")
                        context_parts.append(f"Type: {metadata.get('node_type', 'Unknown')}")
                        context_parts.append(f"Content: {content}")
                        context_parts.append("---")
                        
                    context_text = "\n".join(context_parts)
                    
                    # Build messages for direct LM Studio call - consider model token limits
                    # The model has 4096 tokens, reserve ~500 for system message, response, and overhead
                    # Estimate ~4 characters per token, so max context should be ~14000 characters
                    max_safe_context = 14000
                    user_context_limit = message.context_size
                    
                    # Use the smaller of user preference and safe model limit
                    effective_context_limit = min(user_context_limit, max_safe_context)
                    
                    print(f"📏 Model context limit: {max_safe_context} chars, User requested: {user_context_limit} chars")
                    print(f"📏 Using effective context limit: {effective_context_limit} characters")
                    print(f"📏 Total context length: {len(context_text)} characters")
                    
                    # Add enhanced context (graph query results) if available
                    if enhanced_context_parts:
                        enhanced_context_text = "\n\n".join(enhanced_context_parts)
                        context_text = f"{context_text}\n\n=== GRAPH QUERY RESULTS ===\n{enhanced_context_text}"
                        print(f"📊 Added {len(enhanced_context_parts)} graph query results to context")
                    
                    context_summary = context_text[:effective_context_limit] + "..." if len(context_text) > effective_context_limit else context_text
                    
                    # Build enhanced system message that lets LLM know about its capabilities
                    base_system_msg = """You are a ChatDBT AI assistant - an expert dbt project analyst with access to both vector search and graph query capabilities.

IMPORTANT: You have access to precise graph query results for dependency questions. When you see "GRAPH QUERY RESULT" data in the context, prioritize that information as it's more accurate than vector search for relationships.

Your capabilities:
• Answer questions about dbt models, packages, tests, sources, and relationships
• Analyze dependencies (upstream/downstream) with precise graph data
• Search and list models, packages, and resources
• Provide insights about data lineage and impact analysis
• Explain dbt project structure and relationships

For dependency questions: Use the graph query results when available, as they provide exact relationship data from the dependency graph.
For general questions: Use both vector search context and graph results to provide comprehensive answers.
For package/model listings: Provide complete, accurate lists based on the available data.

Always provide helpful, accurate, and actionable responses."""
                    
                    # Build user message based on query type
                    if intent == 'package_list':
                        user_msg = f"Based on the dbt project data below, please list ALL packages found in this project in a clear, comprehensive format.\n\nData:\n{context_summary}\n\nQuestion: {message.message}"
                    elif intent in ['graph_dependency', 'graph_search', 'graph_commonality']:
                        user_msg = f"I have a question about dbt model relationships/analysis. I've provided both vector search context and precise graph query results below.\n\nPrioritize the GRAPH QUERY RESULT data for dependency/relationship/commonality information as it's more accurate.\n\nData:\n{context_summary}\n\nQuestion: {message.message}"
                    elif intent == 'model_search' and any(pkg in message.message.lower() for pkg in ['package', 'pkg']):
                        user_msg = f"Based on the dbt project data below, please provide a comprehensive list of models for the requested package.\n\nData:\n{context_summary}\n\nQuestion: {message.message}"
                    else:
                        user_msg = f"Based on the dbt project data below, please answer the question accurately and helpfully.\n\nData:\n{context_summary}\n\nQuestion: {message.message}"
                    
                    messages = [
                        {"role": "system", "content": base_system_msg},
                        {"role": "user", "content": user_msg}
                    ]
                    
                    print("✅ Context prepared for direct LM Studio streaming!")
                    print(f"🎯 Using system message type: {'Package Model Search' if (intent == 'model_search' and any(pkg in message.message.lower() for pkg in ['package', 'pkg'])) else intent}")
                    print(f"📝 Context contains: {len(retrieved_nodes)} documents")
                    
                except Exception as context_ex:
                    print(f"❌ Error preparing context: {context_ex}")
                    with request_lock:
                        if request_id in active_requests:
                            active_requests[request_id]['status'] = 'error'
                    raise
                
                # 🚀 DIRECT STREAMING: Stream from LM Studio with TRUE cancellation support
                print("🚀 Starting direct streaming from LM Studio...")
                response_text = ""
                token_count = 0
                
                try:
                    # Use the simple global variable that gets updated by model switching
                    current_model = current_active_model
                    print(f"🔍 DEBUG: Current active model: {current_active_model}")
                    print(f"🔍 DEBUG: Using current_model: {current_model}")
                    print(f"🤖 Using model: {current_model}")
                    
                    # Stream directly from LM Studio with TRUE cancellation
                    # Access the global config object for the streaming function
                    stream_config = {
                        'LM_STUDIO_BASE_URL': config.LM_STUDIO_BASE_URL,
                        'LM_STUDIO_API_KEY': config.LM_STUDIO_API_KEY,
                        'TEMPERATURE': config.TEMPERATURE,
                        'MAX_TOKENS': config.MAX_TOKENS
                    }
                    async for token in stream_from_lm_studio_direct(messages, current_model, request_id, stream_config):
                        # Check for cancellation BEFORE processing token
                        with request_lock:
                            if request_id in active_requests and active_requests[request_id]['cancel_requested']:
                                print(f"🛑 STOP BUTTON: Request cancelled during direct streaming at token {token_count}")
                                active_requests[request_id]['status'] = 'cancelled'
                                yield f"data: {json.dumps({'error': 'Request cancelled by user', 'done': True})}\n\n"
                                return
                        
                        # Check for client disconnect
                        if await request.is_disconnected():
                            print(f"Client disconnected during direct streaming at token {token_count}")
                            with request_lock:
                                if request_id in active_requests:
                                    active_requests[request_id]['status'] = 'disconnected'
                            break
                        
                        response_text += token
                        token_count += 1
                        
                        # Send each token as it arrives
                        chunk = f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                        yield chunk
                        
                        # Small delay to allow disconnect check
                        await asyncio.sleep(0.001)
                        
                except asyncio.CancelledError:
                    print(f"🛑 STOP BUTTON: Direct streaming cancelled for request {request_id}")
                    yield f"data: {json.dumps({'error': 'Request cancelled by user', 'done': True})}\n\n"
                    return
                except Exception as streaming_ex:
                    print(f"❌ Error during direct streaming: {streaming_ex}")
                    print(f"❌ Exception type: {type(streaming_ex)}")
                    import traceback
                    traceback.print_exc()
                    yield f"data: {json.dumps({'error': f'Streaming error: {str(streaming_ex)}', 'done': True})}\n\n"
                    return
                
                # Check if client is still connected before finalizing
                if await request.is_disconnected():
                    print("Client disconnected, skipping finalization")
                    with request_lock:
                        if request_id in active_requests:
                            active_requests[request_id]['status'] = 'disconnected'
                    return
                
                # Final cancellation check
                with request_lock:
                    if request_id in active_requests and active_requests[request_id]['cancel_requested']:
                        print("🛑 Request cancelled during finalization")
                        return
                
                print(f"Completed streaming. Total tokens: {token_count}")
                
                # Extract sources after completion
                sources = []
                try:
                    if hasattr(retrieved_nodes, 'source_nodes') and retrieved_nodes.source_nodes:
                        print(f"Extracting sources from {len(retrieved_nodes.source_nodes)} nodes...")
                        for node in retrieved_nodes.source_nodes:
                            node_name = node.metadata.get('name') or node.metadata.get('node_id', 'Unknown')
                            node_type = node.metadata.get('node_type', 'unknown')
                            file_path = node.metadata.get('file_path')
                            
                            source_ref = f"{node_type}: {node_name}"
                            if file_path:
                                source_ref += f" ({file_path})"
                            
                            sources.append(source_ref)
                    else:
                        print("No source nodes found or source_nodes attribute missing")
                except Exception as source_error:
                    print(f"Error extracting sources: {source_error}")
                
                # Send final message with complete response and sources
                final_chunk = f"data: {json.dumps({'response': response_text, 'sources': sources[:5], 'done': True})}\n\n"
                print("Sending final completion chunk")
                yield final_chunk
                
            except Exception as e:
                print(f"Error in streaming response: {e}")
                import traceback
                traceback.print_exc()
                
                # Update request status
                with request_lock:
                    if request_id in active_requests:
                        active_requests[request_id]['status'] = 'error'
                
                # Check if it's a disconnection error
                if await request.is_disconnected():
                    print("Client disconnected during error handling")
                    return
                
                # Send error response
                yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'})}\n\n"
            finally:
                # Clean up request tracking
                with request_lock:
                    if request_id in active_requests:
                        active_requests[request_id]['status'] = 'completed'
                        print(f"🧹 Cleaned up request {request_id}")
                        # Remove old requests (keep only last 10)
                        if len(active_requests) > 10:
                            oldest_id = min(active_requests.keys(), 
                                          key=lambda x: active_requests[x]['start_time'])
                            del active_requests[oldest_id]
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Transfer-Encoding": "chunked",  # Ensure chunked transfer
            }
        )

    except Exception as e:
        print(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/models")
async def get_available_models():
    """Get list of LOADED models from LM Studio."""
    try:
        # Get models from LM Studio
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{config.LM_STUDIO_BASE_URL}/models") as response:
                if response.status != 200:
                    return {
                        "success": False,
                        "error": f"LM Studio returned status {response.status}",
                        "models": [],
                        "current_model": config.LM_STUDIO_MODEL
                    }
                
                models_data = await response.json()
                all_models = models_data.get('data', [])
                print(f"🔍 All models from LM Studio API: {[m.get('id', 'unknown') for m in all_models]}")
                
                # Test each model to see if it's actually loaded and ready
                loaded_models = []
                for model in all_models:
                    model_id = model.get('id', '')
                    if not model_id:
                        continue
                        
                    print(f"🧪 Testing if model {model_id} is loaded...")
                    try:
                        # Test with a quick request
                        test_payload = {
                            "model": model_id,
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1,
                            "stream": False
                        }
                        async with session.post(
                            f"{config.LM_STUDIO_BASE_URL}/chat/completions",
                            json=test_payload,
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as test_response:
                            if test_response.status == 200:
                                print(f"✅ Model {model_id} is LOADED and ready")
                                loaded_models.append({
                                    'id': model_id,
                                    'object': model.get('object', 'model'),
                                    'owned_by': model.get('owned_by', 'lm-studio')
                                })
                            else:
                                print(f"❌ Model {model_id} is NOT loaded (status: {test_response.status})")
                    except Exception as e:
                        print(f"❌ Model {model_id} test failed: {e}")
                
                print(f"🎯 Final loaded models for frontend: {[m['id'] for m in loaded_models]}")
                return {
                    "success": True,
                    "models": loaded_models,
                    "current_model": config.LM_STUDIO_MODEL
                }
            
    except Exception as e:
        print(f"Error getting models: {e}")
        return {
            "success": False,
            "error": f"Error connecting to LM Studio: {str(e)}",
            "models": [],
            "current_model": config.LM_STUDIO_MODEL
        }

@app.post("/models/{model_id:path}")
async def switch_model(model_id: str):
    """Switch to a different model."""
    global chat_engine, config, current_active_model
    
    try:
        # First, validate that the model is actually loaded in LM Studio
        print(f"🔍 Validating model {model_id} is loaded in LM Studio...")
        async with aiohttp.ClientSession() as session:
            # Try to make a simple request with the model to see if it's loaded
            test_payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "stream": False
            }
            async with session.post(
                f"{config.LM_STUDIO_BASE_URL}/chat/completions",
                json=test_payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Model {model_id} not loaded in LM Studio. Status: {response.status}, Error: {error_text}")
                    return {
                        "success": False,
                        "error": f"Model '{model_id}' is not loaded in LM Studio. Please load it first."
                    }
                print(f"✅ Model {model_id} is loaded and ready in LM Studio")
        
        # Update both the config AND the global current model variable
        old_model = config.LM_STUDIO_MODEL
        config.LM_STUDIO_MODEL = model_id
        current_active_model = model_id  # Update the simple global variable
        
        # Reinitialize the chat engine with the new model
        print(f"Switching from {old_model} to {model_id}")
        
        # Create new LM Studio client with updated model
        lm_studio_llm = LMStudioLLM(
            model=config.LM_STUDIO_MODEL,
            api_base=config.LM_STUDIO_BASE_URL,
            api_key=config.LM_STUDIO_API_KEY,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        
        # Update global settings
        Settings.llm = lm_studio_llm
        
        # Recreate chat engine with new model
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K),
            system_message=(
                "You are an expert analyst of dbt (data build tool) projects. "
                "You have access to a comprehensive knowledge graph of dbt models, sources, tests, and their relationships. "
                "Provide detailed, specific information about data models, transformations, dependencies, and metadata. "
                "When discussing models, always reference actual model names, column names, and relationships from the knowledge graph. "
                "If asked about data lineage or dependencies, explain the full chain of relationships. "
                "For questions about data quality, mention relevant tests and validations. "
                "Always be specific and reference the actual model/table/column names from the knowledge graph."
            )
        )
        
        print(f"Successfully switched to model: {model_id}")
        
        return {
            "success": True,
            "message": f"Successfully switched to model: {model_id}",
            "old_model": old_model,
            "new_model": model_id
        }
        
    except Exception as e:
        # Revert config on error
        config.LM_STUDIO_MODEL = old_model if 'old_model' in locals() else config.LM_STUDIO_MODEL
        
        return {
            "success": False,
            "error": f"Error switching to model {model_id}: {str(e)}",
            "current_model": config.LM_STUDIO_MODEL
        }

@app.post("/query-graph")
async def query_graph(query: dict):
    """Query the graph store for relationships."""
    try:
        import json
        from pathlib import Path
        
        # Load the graph store
        graph_store_path = Path(config.INDEX_STORAGE_PATH) / "graph_store.json"
        if not graph_store_path.exists():
            return {"error": "Graph store not found"}
            
        with open(graph_store_path, 'r') as f:
            graph_data = json.load(f)
        
        graph_dict = graph_data.get('graph_dict', {})
        entities = graph_dict.get('entities', {})
        relations = graph_dict.get('relations', [])
        
        query_type = query.get('type', 'dependencies')
        target_model = query.get('model', '').lower()
        
        if query_type == 'dependencies':
            # Find what depends on the target model
            dependencies = []
            for rel in relations:
                source = rel.get('source', '')
                target = rel.get('target', '')
                
                if target_model in target.lower():
                    dependencies.append({
                        'dependent_model': source,
                        'depends_on': target,
                        'relationship_type': rel.get('label', 'unknown')
                    })
            
            return {
                'query': f"Models that depend on {target_model}",
                'results': dependencies,
                'count': len(dependencies)
            }
        
        elif query_type == 'upstream':
            # Find what the target model depends on
            upstream = []
            for rel in relations:
                source = rel.get('source', '')
                target = rel.get('target', '')
                
                if target_model in source.lower():
                    upstream.append({
                        'model': source,
                        'depends_on': target,
                        'relationship_type': rel.get('label', 'unknown')
                    })
            
            return {
                'query': f"What {target_model} depends on",
                'results': upstream,
                'count': len(upstream)
            }
        
        else:
            return {"error": "Unknown query type"}
            
    except Exception as e:
        return {"error": str(e)}

@app.post("/refresh-index")
async def refresh_index():
    """Refresh the index by reloading knowledge graph data."""
    try:
        global index, chat_engine, graph_helper

        print("Refreshing index...")
        
        # Reset graph helper so it gets reinitialized with fresh data
        graph_helper = None
        
        if config.USE_PROPERTY_GRAPH:
            print("Refreshing Hybrid VectorStore with PropertyGraph data...")
            # Reload dbt artifacts for property graph
            artifacts_loader = DBTArtifactsLoader(config.DBT_ARTIFACTS_PATH)
            documents, entity_nodes, relations = artifacts_loader.load_property_graph_data()

            if not documents:
                raise HTTPException(status_code=400, detail="No documents found in dbt artifacts for refreshing. Please check your DBT_ARTIFACTS_PATH.")

            print(f"Reloaded {len(documents)} documents with {len(entity_nodes)} entities and {len(relations)} relations")
            print("Creating VectorStore index from PropertyGraph documents (true hybrid approach)...")

            # Create VectorStore index with PropertyGraph-derived documents
            # This is the true hybrid approach: vector retrieval + graph-structured content
            index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )
            
            print(f"Hybrid VectorStore index refreshed with {len(documents)} documents")
            print(f"Includes {len(entity_nodes)} entities and {len(relations)} relations from PropertyGraph data")
            
        else:
            print("Refreshing VectorStoreIndex...")
            # Reload dbt artifacts for document creation only
            artifacts_loader = DBTArtifactsLoader(config.DBT_ARTIFACTS_PATH)
            documents = artifacts_loader.load_documents()

            if not documents:
                raise HTTPException(status_code=400, detail="No documents found in dbt artifacts for refreshing. Please check your DBT_ARTIFACTS_PATH.")

            if not all(isinstance(doc, Document) for doc in documents):
                raise TypeError("DBTArtifactsLoader must return a list of llama_index.core.schema.Document objects when refreshing.")

            # Recreate vector index
            index = VectorStoreIndex.from_documents(documents)

        # Persist updated index
        storage_path = Path(config.INDEX_STORAGE_PATH)
        storage_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(storage_path))
        print(f"Index refreshed and persisted to {storage_path}")
        
        # Repopulate graph_store.json AFTER persisting (so it doesn't get overwritten)
        if config.USE_PROPERTY_GRAPH and 'entity_nodes' in locals() and 'relations' in locals():
            print("Repopulating graph_store.json after refresh...")
            from build_index import _populate_graph_store_after_persist
            _populate_graph_store_after_persist(storage_path, entity_nodes, relations)
            print("✅ Graph store repopulated successfully")

        # Recreate chat engine with enhanced capabilities
        if config.USE_PROPERTY_GRAPH:
            print("Creating PropertyGraph-aware chat engine...")
            retriever = index.as_retriever(
                similarity_top_k=config.SIMILARITY_TOP_K,
            )
            
            # Generate dynamic system message based on actual loaded data
            entity_count = len(entity_nodes) if 'entity_nodes' in locals() else "thousands of"
            relation_count = len(relations) if 'relations' in locals() else "thousands of"
            
            chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever,
                system_message=(
                    "You are a dbt project analyst with access to a comprehensive PropertyGraph knowledge base. "
                    "You can query both documents AND graph relationships directly. "
                    "For statistical queries like 'What packages have the most models?', provide exact counts from the graph data. "
                    "For lineage queries, trace actual dependencies through the graph relationships. "
                    "When asked about packages, models, tests, or sources, give specific names and counts. "
                    f"You have access to {entity_count} entities and {relation_count} relationships across multiple dbt packages. "
                    "The graph contains various packages with models, tests, sources, and seeds. "
                    "Always provide precise, data-driven answers with exact statistics when available."
                )
            )
        else:
            chat_engine = CondensePlusContextChatEngine.from_defaults(
                index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K),
                system_message=(
                    "You are a helpful assistant specialized in analyzing dbt (data build tool) projects and knowledge graphs. "
                    "You have access to comprehensive information about data models, their relationships, dependencies, "
                    "SQL transformations, tests, sources, and metadata. "
                    "When answering questions, provide specific information about the models, their relationships, "
                    "and how they transform data. Reference specific model names, dependencies, and SQL logic when relevant. "
                    "If asked about data lineage or dependencies, explain the full chain of relationships. "
                    "For questions about data quality, mention relevant tests and validations. "
                    "Always be specific and reference the actual model/table/column names from the knowledge graph."
                )
            )

        return {"message": f"Index refreshed successfully with {len(documents)} documents."}

    except Exception as e:
        print(f"Error refreshing index: {e}")
        raise HTTPException(status_code=500, detail=f"Error refreshing index: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=config.HOST, 
        port=config.PORT,
        # Disable buffering for real-time streaming
        access_log=True,
        use_colors=True,
        # Important: disable buffering
        loop="asyncio",
        http="h11"
    )
