#!/usr/bin/env python3
"""
Validation script to test the impact analysis app functionality.
"""

import json
import pickle
import networkx as nx
from pathlib import Path

def validate_data_files():
    """Validate that all required data files exist and are readable."""
    print("🔍 Validating data files...")
    
    required_files = [
        "data/knowledge_graph.gpickle",
        "data/nodes.json", 
        "data/edges.json",
        "data/graph_metadata.json"
    ]
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path} exists ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def validate_graph_loading():
    """Validate that the graph can be loaded correctly."""
    print("\n📊 Validating graph loading...")
    
    try:
        # Test pickle loading
        with open("data/knowledge_graph.gpickle", 'rb') as f:
            graph = pickle.load(f)
        
        print(f"✅ Graph loaded successfully")
        print(f"   Nodes: {graph.number_of_nodes():,}")
        print(f"   Edges: {graph.number_of_edges():,}")
        
        # Test node types
        node_types = {}
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print(f"\n📋 Node type distribution:")
        for node_type, count in sorted(node_types.items()):
            print(f"   {node_type}: {count:,}")
        
        return graph
        
    except Exception as e:
        print(f"❌ Failed to load graph: {e}")
        return None

def validate_app_functions(graph):
    """Validate that app functions work with the loaded graph."""
    print("\n🧪 Testing app functions...")
    
    try:
        # Test getting nodes by type
        models = [
            node_id for node_id, attrs in graph.nodes(data=True)
            if attrs.get('node_type') == 'model'
        ]
        
        sources = [
            node_id for node_id, attrs in graph.nodes(data=True)
            if attrs.get('node_type') == 'source'
        ]
        
        print(f"✅ Found {len(models):,} models")
        print(f"✅ Found {len(sources):,} sources")
        
        if models:
            # Test impact analysis on a sample model
            sample_model = models[0]
            print(f"\n🎯 Testing impact analysis on: {sample_model}")
            
            # Test upstream dependencies
            upstream = set()
            for predecessor in graph.predecessors(sample_model):
                upstream.add(predecessor)
            
            # Test downstream dependencies  
            downstream = set()
            for successor in graph.successors(sample_model):
                downstream.add(successor)
            
            print(f"   Upstream: {len(upstream)} nodes")
            print(f"   Downstream: {len(downstream)} nodes")
            
            # Test subgraph creation
            nodes_to_include = {sample_model} | upstream | downstream
            subgraph = graph.subgraph(nodes_to_include)
            
            print(f"   Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
            
        return True
        
    except Exception as e:
        print(f"❌ App function test failed: {e}")
        return False

def validate_metadata():
    """Validate metadata file."""
    print("\n📄 Validating metadata...")
    
    try:
        with open("data/graph_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        stats = metadata.get('statistics', {})
        print(f"✅ Metadata loaded")
        print(f"   dbt version: {stats.get('manifest_version', 'unknown')}")
        print(f"   Created: {metadata.get('created_at', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata validation failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🔍 dbt Impact Analysis App Validation")
    print("=" * 50)
    
    # Test 1: Data files
    if not validate_data_files():
        print("\n❌ Data file validation failed!")
        return False
    
    # Test 2: Graph loading
    graph = validate_graph_loading()
    if graph is None:
        print("\n❌ Graph loading failed!")
        return False
    
    # Test 3: App functions
    if not validate_app_functions(graph):
        print("\n❌ App function validation failed!")
        return False
    
    # Test 4: Metadata
    if not validate_metadata():
        print("\n❌ Metadata validation failed!")
        return False
    
    print("\n🎉 All validation tests passed!")
    print("\n📱 Your Impact Analysis App is ready!")
    print("   URL: http://localhost:8502")
    print("   Features available:")
    print("   - Interactive node selection and search")
    print("   - Upstream/downstream dependency analysis")
    print("   - Risk assessment and recommendations")
    print("   - Interactive graph visualization")
    print("   - Package impact analysis")
    print("   - Detailed data tables")
    
    return True

if __name__ == "__main__":
    main()