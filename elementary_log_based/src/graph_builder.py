"""
Graph builder module for constructing knowledge graphs from Elementary data.
"""
import networkx as nx
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_entities import (
    GraphNode, GraphEdge, NodeFactory, EdgeFactory, 
    NodeType, EdgeType
)
from src.data_extractor import ElementaryDataExtractor


class KnowledgeGraphBuilder:
    """Build knowledge graph from dbt Elementary data."""
    
    def __init__(self):
        """Initialize the graph builder."""
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.stats = defaultdict(int)
    
    def build_graph_from_data(self, data: Dict[str, pd.DataFrame]) -> nx.MultiDiGraph:
        """Build complete knowledge graph from extracted data."""
        print("🏗️  Building knowledge graph...")
        
        # Step 1: Create nodes for each entity type
        self._create_model_nodes(data.get('models', pd.DataFrame()))
        self._create_test_nodes(data.get('tests', pd.DataFrame()))
        self._create_source_nodes(data.get('sources', pd.DataFrame()))
        self._create_column_nodes(data.get('model_columns', pd.DataFrame()))
        self._create_invocation_nodes(data.get('invocations', pd.DataFrame()))
        
        # Step 2: Build edges based on relationships
        self._create_dependency_edges(data.get('models', pd.DataFrame()))
        self._create_test_edges(data.get('tests', pd.DataFrame()))
        self._create_column_edges(data.get('model_columns', pd.DataFrame()))
        self._create_execution_edges(data.get('run_results', pd.DataFrame()))
        self._create_test_result_edges(data.get('test_results', pd.DataFrame()))
        
        # Step 3: Add all nodes and edges to NetworkX graph
        self._build_networkx_graph()
        
        self._print_graph_stats()
        return self.graph
    
    def _create_model_nodes(self, models_df: pd.DataFrame) -> None:
        """Create nodes for dbt models."""
        for _, model in models_df.iterrows():
            node = NodeFactory.create_model_node(model.to_dict())
            self.nodes[node.node_id] = node
            self.stats['model_nodes'] += 1
        
        print(f"📦 Created {len(models_df)} model nodes")
    
    def _create_test_nodes(self, tests_df: pd.DataFrame) -> None:
        """Create nodes for dbt tests."""
        for _, test in tests_df.iterrows():
            node = NodeFactory.create_test_node(test.to_dict())
            self.nodes[node.node_id] = node
            self.stats['test_nodes'] += 1
        
        print(f"🧪 Created {len(tests_df)} test nodes")
    
    def _create_source_nodes(self, sources_df: pd.DataFrame) -> None:
        """Create nodes for dbt sources."""
        for _, source in sources_df.iterrows():
            node = NodeFactory.create_source_node(source.to_dict())
            self.nodes[node.node_id] = node
            self.stats['source_nodes'] += 1
        
        print(f"📊 Created {len(sources_df)} source nodes")
    
    def _create_column_nodes(self, columns_df: pd.DataFrame) -> None:
        """Create nodes for model columns."""
        for _, column in columns_df.iterrows():
            model_id = column['model_unique_id']
            if model_id in self.nodes:  # Only create column if model exists
                node = NodeFactory.create_column_node(column.to_dict(), model_id)
                self.nodes[node.node_id] = node
                self.stats['column_nodes'] += 1
        
        print(f"🏛️ Created {self.stats['column_nodes']} column nodes")
    
    def _create_invocation_nodes(self, invocations_df: pd.DataFrame) -> None:
        """Create nodes for dbt invocations."""
        for _, invocation in invocations_df.iterrows():
            node = NodeFactory.create_invocation_node(invocation.to_dict())
            self.nodes[node.node_id] = node
            self.stats['invocation_nodes'] += 1
        
        print(f"🚀 Created {len(invocations_df)} invocation nodes")
    
    def _create_dependency_edges(self, models_df: pd.DataFrame) -> None:
        """Create dependency edges between models."""
        for _, model in models_df.iterrows():
            edges = EdgeFactory.create_dependency_edges(model.to_dict())
            for edge in edges:
                # Only add edge if both nodes exist
                if edge.source_id in self.nodes and edge.target_id in self.nodes:
                    self.edges.append(edge)
                    self.stats['dependency_edges'] += 1
        
        print(f"🔗 Created {self.stats['dependency_edges']} dependency edges")
    
    def _create_test_edges(self, tests_df: pd.DataFrame) -> None:
        """Create test edges between tests and tested nodes."""
        for _, test in tests_df.iterrows():
            edges = EdgeFactory.create_test_edges(test.to_dict())
            for edge in edges:
                # Only add edge if both nodes exist
                if edge.source_id in self.nodes and edge.target_id in self.nodes:
                    self.edges.append(edge)
                    self.stats['test_edges'] += 1
        
        print(f"🧪 Created {self.stats['test_edges']} test edges")
    
    def _create_column_edges(self, columns_df: pd.DataFrame) -> None:
        """Create edges between models and their columns."""
        for _, column in columns_df.iterrows():
            model_id = column['model_unique_id']
            column_id = f"{model_id}::{column['column_name']}"
            
            if model_id in self.nodes and column_id in self.nodes:
                edge = EdgeFactory.create_column_edges(column.to_dict(), model_id)
                self.edges.append(edge)
                self.stats['column_edges'] += 1
        
        print(f"🏛️ Created {self.stats['column_edges']} column edges")
    
    def _create_execution_edges(self, run_results_df: pd.DataFrame) -> None:
        """Create execution edges between invocations and models."""
        for _, result in run_results_df.iterrows():
            invocation_id = result['invocation_id']
            model_id = result['model_unique_id']
            
            if invocation_id in self.nodes and model_id in self.nodes:
                edge = EdgeFactory.create_execution_edges(result.to_dict())
                self.edges.append(edge)
                self.stats['execution_edges'] += 1
        
        print(f"⚡ Created {self.stats['execution_edges']} execution edges")
    
    def _create_test_result_edges(self, test_results_df: pd.DataFrame) -> None:
        """Create test result edges between invocations and tests."""
        for _, result in test_results_df.iterrows():
            invocation_id = result['invocation_id']
            test_id = result['test_unique_id']
            
            if invocation_id in self.nodes and test_id in self.nodes:
                edge = EdgeFactory.create_test_result_edges(result.to_dict())
                self.edges.append(edge)
                self.stats['test_result_edges'] += 1
        
        print(f"📋 Created {self.stats['test_result_edges']} test result edges")
    
    def _build_networkx_graph(self) -> None:
        """Add all nodes and edges to the NetworkX graph."""
        # Add nodes with their attributes
        for node in self.nodes.values():
            self.graph.add_node(
                node.node_id,
                name=node.name,
                node_type=node.node_type.value,
                **node.properties
            )
        
        # Add edges with their attributes
        for edge in self.edges:
            self.graph.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type.value,
                **edge.properties
            )
        
        print(f"🎯 Added {len(self.nodes)} nodes and {len(self.edges)} edges to NetworkX graph")
    
    def _print_graph_stats(self) -> None:
        """Print comprehensive graph statistics."""
        print("\n📊 Knowledge Graph Statistics:")
        print("=" * 40)
        
        # Node statistics
        print("Nodes:")
        for node_type in NodeType:
            count = self.stats.get(f'{node_type.value}_nodes', 0)
            print(f"  - {node_type.value.title()}: {count}")
        
        print(f"  Total Nodes: {len(self.nodes)}")
        
        # Edge statistics
        print("\nEdges:")
        for edge_type in EdgeType:
            count = self.stats.get(f'{edge_type.value}_edges', 0)
            print(f"  - {edge_type.value.replace('_', ' ').title()}: {count}")
        
        print(f"  Total Edges: {len(self.edges)}")
        
        # Graph metrics
        if len(self.graph) > 0:
            print(f"\nGraph Metrics:")
            print(f"  - Density: {nx.density(self.graph):.4f}")
            print(f"  - Connected Components: {nx.number_weakly_connected_components(self.graph)}")
            
            # Degree statistics
            degrees = dict(self.graph.degree())
            if degrees:
                avg_degree = sum(degrees.values()) / len(degrees)
                max_degree = max(degrees.values())
                print(f"  - Average Degree: {avg_degree:.2f}")
                print(f"  - Max Degree: {max_degree}")
    
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[GraphEdge]:
        """Get all edges of a specific type."""
        return [edge for edge in self.edges if edge.edge_type == edge_type]
    
    def find_node_neighbors(self, node_id: str, direction: str = 'both') -> List[str]:
        """Find neighbors of a node."""
        if node_id not in self.graph:
            return []
        
        if direction == 'incoming':
            return list(self.graph.predecessors(node_id))
        elif direction == 'outgoing':
            return list(self.graph.successors(node_id))
        else:  # both
            predecessors = set(self.graph.predecessors(node_id))
            successors = set(self.graph.successors(node_id))
            return list(predecessors.union(successors))
    
    def get_model_lineage(self, model_id: str, depth: int = None) -> Dict[str, List[str]]:
        """Get upstream and downstream lineage for a model."""
        if model_id not in self.graph:
            return {'upstream': [], 'downstream': []}
        
        # Get upstream dependencies (predecessors)
        if depth is None:
            upstream = list(nx.ancestors(self.graph, model_id))
        else:
            upstream = []
            current_level = {model_id}
            for _ in range(depth):
                next_level = set()
                for node in current_level:
                    predecessors = set(self.graph.predecessors(node))
                    next_level.update(predecessors)
                    upstream.extend(predecessors)
                current_level = next_level
                if not current_level:
                    break
        
        # Get downstream dependencies (successors)
        if depth is None:
            downstream = list(nx.descendants(self.graph, model_id))
        else:
            downstream = []
            current_level = {model_id}
            for _ in range(depth):
                next_level = set()
                for node in current_level:
                    successors = set(self.graph.successors(node))
                    next_level.update(successors)
                    downstream.extend(successors)
                current_level = next_level
                if not current_level:
                    break
        
        return {
            'upstream': list(set(upstream)),
            'downstream': list(set(downstream))
        }
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph data for external use."""
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges],
            'stats': dict(self.stats)
        }