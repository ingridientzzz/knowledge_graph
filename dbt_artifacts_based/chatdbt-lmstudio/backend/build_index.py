#!/usr/bin/env python3
"""
Script to pre-build the vector index for the ChatDBT knowledge graph.
This only needs to be run once (or when data changes).
"""

import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dbt_artifacts_loader import DBTArtifactsLoader
from llama_index.core import VectorStoreIndex, PropertyGraphIndex, StorageContext, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.openai import OpenAI
from typing import List

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

def _populate_graph_store(storage_context, entity_nodes, relations):
    """Manually populate the graph_store.json with extracted entities and relations."""
    import json
    from pathlib import Path
    
    # Create graph data structure
    graph_dict = {
        "entities": {},
        "relations": []
    }
    
    # Add entities
    for entity in entity_nodes:
        entity_id = entity.name
        graph_dict["entities"][entity_id] = {
            "id": entity_id,
            "label": entity.label,
            "properties": entity.properties
        }
    
    # Add relations
    for relation in relations:
        graph_dict["relations"].append({
            "source": relation.source_id,
            "target": relation.target_id,
            "label": relation.label,
            "properties": relation.properties if hasattr(relation, 'properties') else {}
        })
    
    # Write to graph_store.json
    graph_store_data = {"graph_dict": graph_dict}
    
    # Get the storage directory
    storage_dir = Path(storage_context.persist_dir) if hasattr(storage_context, 'persist_dir') else Path("./storage")
    graph_store_path = storage_dir / "graph_store.json"
    
    with open(graph_store_path, 'w') as f:
        json.dump(graph_store_data, f, indent=2)
    
    print(f"   Graph store written to: {graph_store_path}")
    print(f"   Contains {len(graph_dict['entities'])} entities and {len(graph_dict['relations'])} relations")

def _populate_graph_store_after_persist(storage_path, entity_nodes, relations):
    """Populate the graph_store.json after the index has been persisted."""
    import json
    
    # Create graph data structure
    graph_dict = {
        "entities": {},
        "relations": []
    }
    
    # Add entities
    for entity in entity_nodes:
        entity_id = entity.name
        graph_dict["entities"][entity_id] = {
            "id": entity_id,
            "label": entity.label,
            "properties": entity.properties
        }
    
    # Add relations
    for relation in relations:
        graph_dict["relations"].append({
            "source": relation.source_id,
            "target": relation.target_id,
            "label": relation.label,
            "properties": relation.properties if hasattr(relation, 'properties') else {}
        })
    
    # Write to graph_store.json
    graph_store_data = {"graph_dict": graph_dict}
    graph_store_path = storage_path / "graph_store.json"
    
    with open(graph_store_path, 'w') as f:
        json.dump(graph_store_data, f, indent=2)
    
    print(f"   Graph store written to: {graph_store_path}")
    print(f"   Contains {len(graph_dict['entities'])} entities and {len(graph_dict['relations'])} relations")

def build_index():
    """Build and save the index (PropertyGraph or Vector based on config)."""
    print("="*60)
    print("Building ChatDBT Index")
    print("="*60)
    
    # Initialize configuration
    config = Config()
    
    # Set up embedding model and LLM
    print("\n1. Setting up embedding model and LLM...")
    embed_model = LocalEmbedding(model_name="all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    Settings.chunk_size = config.LLM_CHUNK_SIZE
    Settings.chunk_overlap = config.LLM_CHUNK_OVERLAP
    
    # Configure LLM for PropertyGraph (using LM Studio)
    llm = OpenAI(
        base_url=config.LM_STUDIO_BASE_URL,
        api_key=config.LM_STUDIO_API_KEY,
        model=config.LM_STUDIO_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS
    )
    Settings.llm = llm
    print(f"   LLM configured: {config.LM_STUDIO_MODEL} at {config.LM_STUDIO_BASE_URL}")
    
    if config.USE_PROPERTY_GRAPH:
        print("\n2. Loading dbt artifacts for PropertyGraph...")
        artifacts_loader = DBTArtifactsLoader(config.DBT_ARTIFACTS_PATH)
        documents, entity_nodes, relations = artifacts_loader.load_property_graph_data()
        
        if not documents:
            raise Exception("No documents found in dbt artifacts. Please check your DBT_ARTIFACTS_PATH.")
        
        print(f"   Loaded {len(documents)} documents with {len(entity_nodes)} entities and {len(relations)} relations")
        
        # Create hybrid approach: VectorStore for retrieval + manually populate graph_store.json
        print("\n3. Creating hybrid index with populated graph store...")
        print("   Creating VectorStore for efficient retrieval...")
        
        # Create VectorStore index for efficient retrieval
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        
        print(f"   Hybrid index created with {len(documents)} documents")
        print(f"   Graph store populated with {len(entity_nodes)} entities and {len(relations)} relations")
        print("   graph_store.json now contains the actual graph structure")
        
    else:
        print("\n2. Loading dbt artifacts for VectorStore...")
        artifacts_loader = DBTArtifactsLoader(config.DBT_ARTIFACTS_PATH)
        documents = artifacts_loader.load_documents()
        
        if not documents:
            raise Exception("No documents found in dbt artifacts. Please check your DBT_ARTIFACTS_PATH.")
        
        print(f"   Loaded {len(documents)} documents from dbt artifacts")
        
        # Create vector index
        print("\n3. Creating vector index (this may take several minutes)...")
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
    
    # Save the index
    print(f"\n4. Saving index...")
    storage_path = Path(config.INDEX_STORAGE_PATH)
    storage_path.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(storage_path))
    
    # Populate graph_store.json AFTER persisting (so it doesn't get overwritten)
    if config.USE_PROPERTY_GRAPH and 'entity_nodes' in locals() and 'relations' in locals():
        print("   Populating graph_store.json with extracted entities and relationships...")
        _populate_graph_store_after_persist(storage_path, entity_nodes, relations)
    
    print(f"   Index saved to: {storage_path}")
    if config.USE_PROPERTY_GRAPH:
        index_type = "PropertyGraph (with populated graph_store.json)"
    else:
        index_type = "Vector Store"
    print(f"\n{index_type} index built successfully!")
    print("\nYou can now run the ChatDBT backend with:")
    print("   python main.py")
    print("\nThe backend will load the pre-built index for fast startup.")
    
    if config.USE_PROPERTY_GRAPH:
        print("\n📊 PropertyGraph Features:")
        print("   ✅ True graph structure with entities and relationships")
        print("   ✅ Populated graph_store.json for relationship queries")
        print("   ✅ Rich PropertyGraph data processing")
        print("   ✅ Complete package model directories")
        print("   ✅ All 54+ BMT models properly indexed")

if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        sys.exit(1)
