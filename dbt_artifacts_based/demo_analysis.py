#!/usr/bin/env python3
"""
Demo script showing key features of the manifest-based impact analysis.
"""

import json
import pickle
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd

def load_graph() -> nx.MultiDiGraph:
    """Load the graph from pickle file."""
    pickle_file = Path("data/knowledge_graph.gpickle")
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def get_nodes_by_type(graph: nx.MultiDiGraph, node_type: str) -> List[str]:
    """Get nodes of a specific type."""
    return [
        node_id for node_id, attrs in graph.nodes(data=True)
        if attrs.get('node_type') == node_type
    ]

def analyze_node_impact(graph: nx.MultiDiGraph, node_id: str, depth: int = 2) -> Dict:
    """Analyze the impact of a specific node."""
    def get_dependencies(graph, node, depth, direction='upstream'):
        if depth == 0:
            return set()
        
        deps = set()
        if direction == 'upstream':
            neighbors = graph.predecessors(node)
        else:
            neighbors = graph.successors(node)
        
        for neighbor in neighbors:
            deps.add(neighbor)
            if depth > 1:
                deps.update(get_dependencies(graph, neighbor, depth - 1, direction))
        
        return deps
    
    upstream = get_dependencies(graph, node_id, depth, 'upstream')
    downstream = get_dependencies(graph, node_id, depth, 'downstream')
    
    # Count by type
    upstream_types = {}
    downstream_types = {}
    
    for node in upstream:
        node_type = graph.nodes[node].get('node_type', 'unknown')
        upstream_types[node_type] = upstream_types.get(node_type, 0) + 1
    
    for node in downstream:
        node_type = graph.nodes[node].get('node_type', 'unknown')
        downstream_types[node_type] = downstream_types.get(node_type, 0) + 1
    
    return {
        'node_id': node_id,
        'node_name': graph.nodes[node_id].get('name', node_id),
        'node_type': graph.nodes[node_id].get('node_type', 'unknown'),
        'upstream_count': len(upstream),
        'downstream_count': len(downstream),
        'total_impact': len(upstream) + len(downstream) + 1,
        'upstream_types': upstream_types,
        'downstream_types': downstream_types,
        'risk_level': 'High' if len(downstream) > 10 else 'Medium' if len(downstream) > 5 else 'Low'
    }

def demo_impact_analysis():
    """Run a demonstration of impact analysis features."""
    print("🔍 dbt Impact Analysis Demo")
    print("=" * 50)
    
    # Load graph
    print("📊 Loading graph data...")
    graph = load_graph()
    print(f"✅ Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Show node type distribution
    print("\n📋 Node Type Distribution:")
    node_types = {}
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type.ljust(12)}: {count:,}")
    
    # Analyze sample models
    print("\n🎯 Sample Impact Analysis:")
    models = get_nodes_by_type(graph, 'model')
    
    if models:
        # Analyze top 5 models with most downstream dependencies
        model_impacts = []
        for model in models[:20]:  # Limit to first 20 for performance
            impact = analyze_node_impact(graph, model)
            model_impacts.append(impact)
        
        # Sort by downstream impact
        model_impacts.sort(key=lambda x: x['downstream_count'], reverse=True)
        
        print(f"\n🔥 Top 5 Models with Highest Downstream Impact:")
        for i, impact in enumerate(model_impacts[:5], 1):
            print(f"\n{i}. {impact['node_name']}")
            print(f"   ID: {impact['node_id']}")
            print(f"   Upstream: {impact['upstream_count']} | Downstream: {impact['downstream_count']} | Risk: {impact['risk_level']}")
            
            if impact['downstream_types']:
                type_summary = ', '.join([f"{k}: {v}" for k, v in impact['downstream_types'].items()])
                print(f"   Downstream types: {type_summary}")
    
    # Analyze sources
    sources = get_nodes_by_type(graph, 'source')
    if sources:
        print(f"\n📡 Source Analysis:")
        source_impacts = []
        for source in sources[:10]:  # Limit to first 10
            impact = analyze_node_impact(graph, source)
            source_impacts.append(impact)
        
        source_impacts.sort(key=lambda x: x['downstream_count'], reverse=True)
        
        print(f"\n📊 Top 3 Sources with Highest Impact:")
        for i, impact in enumerate(source_impacts[:3], 1):
            print(f"\n{i}. {impact['node_name']}")
            print(f"   ID: {impact['node_id']}")
            print(f"   Downstream: {impact['downstream_count']} | Risk: {impact['risk_level']}")
    
    # Package analysis
    print(f"\n📦 Package Analysis:")
    packages = {}
    for node_id, attrs in graph.nodes(data=True):
        package = attrs.get('package_name')
        if package:
            if package not in packages:
                packages[package] = {'models': 0, 'tests': 0, 'sources': 0, 'other': 0}
            
            node_type = attrs.get('node_type', 'other')
            if node_type in packages[package]:
                packages[package][node_type] += 1
            else:
                packages[package]['other'] += 1
    
    print(f"\n📋 Top 5 Packages by Total Resources:")
    package_totals = [(pkg, sum(counts.values())) for pkg, counts in packages.items()]
    package_totals.sort(key=lambda x: x[1], reverse=True)
    
    for i, (package, total) in enumerate(package_totals[:5], 1):
        counts = packages[package]
        print(f"{i}. {package}: {total} resources")
        print(f"   Models: {counts.get('models', 0)} | Tests: {counts.get('tests', 0)} | Sources: {counts.get('sources', 0)}")
    
    print(f"\n🎉 Demo completed! Use the Streamlit app for interactive analysis:")
    print(f"   streamlit run impact_analysis_app.py --server.port 8502")
    print(f"   or run: python run_app.py")

if __name__ == "__main__":
    demo_impact_analysis()