#!/usr/bin/env python3
"""
Comprehensive manifest.json parser for dbt projects.
Transforms dbt metadata into a graph-friendly format with nodes, relationships, and properties.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import networkx as nx


class NodeType(Enum):
    """Enumeration of node types in the knowledge graph."""
    MODEL = "model"
    TEST = "test"
    SOURCE = "source"
    COLUMN = "column"
    EXPOSURE = "exposure"
    METRIC = "metric"
    PACKAGE = "package"
    MACRO = "macro"
    SEED = "seed"
    SNAPSHOT = "snapshot"
    ANALYSIS = "analysis"


class EdgeType(Enum):
    """Enumeration of edge types in the knowledge graph."""
    DEPENDS_ON = "depends_on"
    HAS_COLUMN = "has_column"
    TESTS = "tests"
    SELECTS_FROM = "selects_from"
    USES = "uses"
    DERIVES_FROM = "derives_from"
    PART_OF_PACKAGE = "part_of_package"
    EXPOSES = "exposes"
    MEASURES = "measures"


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    unique_id: str
    node_type: NodeType
    name: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        result = {
            'id': self.unique_id,
            'node_type': self.node_type.value,
            'name': self.name,
        }
        result.update(self.properties)
        return result


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        result = {
            'source': self.source_id,
            'target': self.target_id,
            'edge_type': self.edge_type.value,
        }
        result.update(self.properties)
        return result


class ManifestParser:
    """Parser for dbt manifest.json files with integrated storage capabilities."""
    
    def __init__(self, manifest_path: str, storage_dir: str = "data"):
        """Initialize the parser with a manifest file path and storage directory."""
        self.manifest_path = Path(manifest_path)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.manifest_data: Dict[str, Any] = {}
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.graph = nx.MultiDiGraph()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # File paths for storage
        self.nodes_file = self.storage_dir / "nodes.json"
        self.edges_file = self.storage_dir / "edges.json"
        self.graph_file = self.storage_dir / "knowledge_graph.gpickle"
        self.metadata_file = self.storage_dir / "graph_metadata.json"
        
    def parse_manifest(self) -> Tuple[Dict[str, GraphNode], List[GraphEdge]]:
        """
        Main parsing method that coordinates all parsing steps.
        Returns nodes and edges dictionaries.
        """
        self.logger.info(f"Starting to parse manifest: {self.manifest_path}")
        
        # Step 1: Parse manifest.json
        self._load_manifest()
        
        # Step 2: Transform metadata into graph-friendly format
        self._identify_nodes()
        self._create_properties()
        self._define_relationships()
        
        self.logger.info(f"Parsing complete. Found {len(self.nodes)} nodes and {len(self.edges)} edges")
        return self.nodes, self.edges
    
    def _load_manifest(self) -> None:
        """Load and parse the manifest.json file."""
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                self.manifest_data = json.load(f)
            self.logger.info(f"Successfully loaded manifest version: {self.manifest_data.get('metadata', {}).get('dbt_version', 'unknown')}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in manifest file: {e}")
    
    def _identify_nodes(self) -> None:
        """Identify and map each resource type to specific node types."""
        self.logger.info("Identifying nodes from manifest...")
        
        # Process nodes section (models, tests, seeds, snapshots, analyses)
        nodes_section = self.manifest_data.get('nodes', {})
        for unique_id, node_data in nodes_section.items():
            resource_type = node_data.get('resource_type')
            
            if resource_type == 'model':
                self._create_model_node(unique_id, node_data)
            elif resource_type == 'test':
                self._create_test_node(unique_id, node_data)
            elif resource_type == 'seed':
                self._create_seed_node(unique_id, node_data)
            elif resource_type == 'snapshot':
                self._create_snapshot_node(unique_id, node_data)
            elif resource_type == 'analysis':
                self._create_analysis_node(unique_id, node_data)
        
        # Process sources section
        sources_section = self.manifest_data.get('sources', {})
        for unique_id, source_data in sources_section.items():
            self._create_source_node(unique_id, source_data)
        
        # Process exposures section
        exposures_section = self.manifest_data.get('exposures', {})
        for unique_id, exposure_data in exposures_section.items():
            self._create_exposure_node(unique_id, exposure_data)
        
        # Process metrics section
        metrics_section = self.manifest_data.get('metrics', {})
        for unique_id, metric_data in metrics_section.items():
            self._create_metric_node(unique_id, metric_data)
        
        # Process macros section
        macros_section = self.manifest_data.get('macros', {})
        for unique_id, macro_data in macros_section.items():
            self._create_macro_node(unique_id, macro_data)
        
        # Create package nodes
        self._create_package_nodes()
        
        # Create column nodes for models and sources
        self._create_column_nodes()
    
    def _create_model_node(self, unique_id: str, node_data: Dict[str, Any]) -> None:
        """Create a model node."""
        properties = {
            'description': node_data.get('description', ''),
            'database': node_data.get('database'),
            'schema': node_data.get('schema'),
            'alias': node_data.get('alias'),
            'package_name': node_data.get('package_name'),
            'path': node_data.get('original_file_path'),
            'materialization': node_data.get('config', {}).get('materialized'),
            'tags': node_data.get('tags', []),
            'meta': node_data.get('meta', {}),
            'checksum': node_data.get('checksum', {}).get('checksum'),
            'resource_type': node_data.get('resource_type'),
            'depends_on_nodes': node_data.get('depends_on', {}).get('nodes', []),
            'compiled_code': node_data.get('compiled_code'),
            'raw_code': node_data.get('raw_code'),
        }
        
        node = GraphNode(
            unique_id=unique_id,
            node_type=NodeType.MODEL,
            name=node_data.get('name', ''),
            properties=properties
        )
        self.nodes[unique_id] = node
    
    def _create_test_node(self, unique_id: str, node_data: Dict[str, Any]) -> None:
        """Create a test node."""
        properties = {
            'description': node_data.get('description', ''),
            'package_name': node_data.get('package_name'),
            'path': node_data.get('original_file_path'),
            'test_metadata': node_data.get('test_metadata', {}),
            'tags': node_data.get('tags', []),
            'meta': node_data.get('meta', {}),
            'resource_type': node_data.get('resource_type'),
            'depends_on_nodes': node_data.get('depends_on', {}).get('nodes', []),
            'compiled_code': node_data.get('compiled_code'),
            'raw_code': node_data.get('raw_code'),
        }
        
        node = GraphNode(
            unique_id=unique_id,
            node_type=NodeType.TEST,
            name=node_data.get('name', ''),
            properties=properties
        )
        self.nodes[unique_id] = node
    
    def _create_source_node(self, unique_id: str, source_data: Dict[str, Any]) -> None:
        """Create a source node."""
        properties = {
            'description': source_data.get('description', ''),
            'database': source_data.get('database'),
            'schema': source_data.get('schema'),
            'source_name': source_data.get('source_name'),
            'table_name': source_data.get('name'),
            'package_name': source_data.get('package_name'),
            'path': source_data.get('original_file_path'),
            'tags': source_data.get('tags', []),
            'meta': source_data.get('meta', {}),
            'resource_type': source_data.get('resource_type'),
            'freshness': source_data.get('freshness', {}),
            'loaded_at_field': source_data.get('loaded_at_field'),
        }
        
        # For source nodes, combine source_name and table_name for the node name
        # e.g., for "source.project.account_optimized.account_edp_optimized"
        # the name should be "account_optimized.account_edp_optimized"
        source_name = source_data.get('source_name') or ''
        table_name = source_data.get('name') or ''
        node_name = f"{source_name}.{table_name}" if source_name and table_name else (table_name or source_name or '')
        
        node = GraphNode(
            unique_id=unique_id,
            node_type=NodeType.SOURCE,
            name=node_name,
            properties=properties
        )
        self.nodes[unique_id] = node
    
    def _create_exposure_node(self, unique_id: str, exposure_data: Dict[str, Any]) -> None:
        """Create an exposure node."""
        properties = {
            'description': exposure_data.get('description', ''),
            'type': exposure_data.get('type'),
            'url': exposure_data.get('url'),
            'owner': exposure_data.get('owner', {}),
            'package_name': exposure_data.get('package_name'),
            'path': exposure_data.get('original_file_path'),
            'tags': exposure_data.get('tags', []),
            'meta': exposure_data.get('meta', {}),
            'resource_type': exposure_data.get('resource_type'),
            'depends_on_nodes': exposure_data.get('depends_on', {}).get('nodes', []),
        }
        
        node = GraphNode(
            unique_id=unique_id,
            node_type=NodeType.EXPOSURE,
            name=exposure_data.get('name', ''),
            properties=properties
        )
        self.nodes[unique_id] = node
    
    def _create_metric_node(self, unique_id: str, metric_data: Dict[str, Any]) -> None:
        """Create a metric node."""
        properties = {
            'description': metric_data.get('description', ''),
            'type': metric_data.get('type'),
            'sql': metric_data.get('sql'),
            'package_name': metric_data.get('package_name'),
            'path': metric_data.get('original_file_path'),
            'tags': metric_data.get('tags', []),
            'meta': metric_data.get('meta', {}),
            'resource_type': metric_data.get('resource_type'),
            'depends_on_nodes': metric_data.get('depends_on', {}).get('nodes', []),
            'model': metric_data.get('model'),
            'dimensions': metric_data.get('dimensions', []),
            'filters': metric_data.get('filters', []),
        }
        
        node = GraphNode(
            unique_id=unique_id,
            node_type=NodeType.METRIC,
            name=metric_data.get('name', ''),
            properties=properties
        )
        self.nodes[unique_id] = node
    
    def _create_seed_node(self, unique_id: str, node_data: Dict[str, Any]) -> None:
        """Create a seed node."""
        properties = {
            'description': node_data.get('description', ''),
            'database': node_data.get('database'),
            'schema': node_data.get('schema'),
            'alias': node_data.get('alias'),
            'package_name': node_data.get('package_name'),
            'path': node_data.get('original_file_path'),
            'tags': node_data.get('tags', []),
            'meta': node_data.get('meta', {}),
            'resource_type': node_data.get('resource_type'),
        }
        
        node = GraphNode(
            unique_id=unique_id,
            node_type=NodeType.SEED,
            name=node_data.get('name', ''),
            properties=properties
        )
        self.nodes[unique_id] = node
    
    def _create_snapshot_node(self, unique_id: str, node_data: Dict[str, Any]) -> None:
        """Create a snapshot node."""
        properties = {
            'description': node_data.get('description', ''),
            'database': node_data.get('database'),
            'schema': node_data.get('schema'),
            'alias': node_data.get('alias'),
            'package_name': node_data.get('package_name'),
            'path': node_data.get('original_file_path'),
            'tags': node_data.get('tags', []),
            'meta': node_data.get('meta', {}),
            'resource_type': node_data.get('resource_type'),
            'strategy': node_data.get('config', {}).get('strategy'),
            'target_schema': node_data.get('config', {}).get('target_schema'),
            'target_database': node_data.get('config', {}).get('target_database'),
        }
        
        node = GraphNode(
            unique_id=unique_id,
            node_type=NodeType.SNAPSHOT,
            name=node_data.get('name', ''),
            properties=properties
        )
        self.nodes[unique_id] = node
    
    def _create_analysis_node(self, unique_id: str, node_data: Dict[str, Any]) -> None:
        """Create an analysis node."""
        properties = {
            'description': node_data.get('description', ''),
            'package_name': node_data.get('package_name'),
            'path': node_data.get('original_file_path'),
            'tags': node_data.get('tags', []),
            'meta': node_data.get('meta', {}),
            'resource_type': node_data.get('resource_type'),
            'depends_on_nodes': node_data.get('depends_on', {}).get('nodes', []),
            'compiled_code': node_data.get('compiled_code'),
            'raw_code': node_data.get('raw_code'),
        }
        
        node = GraphNode(
            unique_id=unique_id,
            node_type=NodeType.ANALYSIS,
            name=node_data.get('name', ''),
            properties=properties
        )
        self.nodes[unique_id] = node
    
    def _create_macro_node(self, unique_id: str, macro_data: Dict[str, Any]) -> None:
        """Create a macro node."""
        properties = {
            'description': macro_data.get('description', ''),
            'package_name': macro_data.get('package_name'),
            'path': macro_data.get('original_file_path'),
            'resource_type': macro_data.get('resource_type'),
            'macro_sql': macro_data.get('macro_sql'),
            'arguments': macro_data.get('arguments', []),
            'meta': macro_data.get('meta', {}),
        }
        
        node = GraphNode(
            unique_id=unique_id,
            node_type=NodeType.MACRO,
            name=macro_data.get('name', ''),
            properties=properties
        )
        self.nodes[unique_id] = node
    
    def _create_package_nodes(self) -> None:
        """Create package nodes based on unique package names."""
        packages = set()
        
        # Collect all unique package names
        for node in self.nodes.values():
            package_name = node.properties.get('package_name')
            if package_name:
                packages.add(package_name)
        
        # Create package nodes
        for package_name in packages:
            unique_id = f"package.{package_name}"
            properties = {
                'resource_type': 'package',
                'description': f"dbt package: {package_name}",
            }
            
            node = GraphNode(
                unique_id=unique_id,
                node_type=NodeType.PACKAGE,
                name=package_name,
                properties=properties
            )
            self.nodes[unique_id] = node
    
    def _create_column_nodes(self) -> None:
        """Create column nodes for models and sources."""
        # Create a list of nodes to avoid modifying dictionary during iteration
        nodes_list = list(self.nodes.items())
        
        for unique_id, node in nodes_list:
            if node.node_type in [NodeType.MODEL, NodeType.SOURCE]:
                # Get column information from manifest
                if node.node_type == NodeType.MODEL:
                    columns_data = self.manifest_data.get('nodes', {}).get(unique_id, {}).get('columns', {})
                else:  # SOURCE
                    columns_data = self.manifest_data.get('sources', {}).get(unique_id, {}).get('columns', {})
                
                for column_name, column_info in columns_data.items():
                    column_unique_id = f"{unique_id}.{column_name}"
                    
                    properties = {
                        'description': column_info.get('description', ''),
                        'data_type': column_info.get('data_type'),
                        'meta': column_info.get('meta', {}),
                        'tags': column_info.get('tags', []),
                        'parent_node': unique_id,
                        'resource_type': 'column',
                    }
                    
                    column_node = GraphNode(
                        unique_id=column_unique_id,
                        node_type=NodeType.COLUMN,
                        name=column_name,
                        properties=properties
                    )
                    self.nodes[column_unique_id] = column_node
    
    def _create_properties(self) -> None:
        """Create and populate properties for nodes."""
        self.logger.info("Creating properties for nodes...")
        
        # Properties are already created in the node creation methods
        # This method can be used for additional property processing
        
        # Process custom meta properties
        nodes_list = list(self.nodes.values())
        for node in nodes_list:
            meta = node.properties.get('meta', {})
            if meta:
                # Add meta properties as direct node properties
                for key, value in meta.items():
                    # Prefix meta properties to avoid conflicts
                    meta_key = f"meta_{key}"
                    node.properties[meta_key] = value
    
    def _define_relationships(self) -> None:
        """Define relationships between nodes based on dependencies and connections."""
        self.logger.info("Defining relationships between nodes...")
        
        # Model dependencies
        self._create_dependency_relationships()
        
        # Column relationships
        self._create_column_relationships()
        
        # Package relationships
        self._create_package_relationships()
        
        # Test relationships
        self._create_test_relationships()
        
        # Exposure relationships
        self._create_exposure_relationships()
        
        # Metric relationships
        self._create_metric_relationships()
    
    def _create_dependency_relationships(self) -> None:
        """Create DEPENDS_ON relationships based on depends_on.nodes."""
        for node in self.nodes.values():
            depends_on_nodes = node.properties.get('depends_on_nodes', [])
            
            for dependency_id in depends_on_nodes:
                if dependency_id in self.nodes:
                    # Determine the relationship type based on the target node type
                    target_node = self.nodes[dependency_id]
                    
                    if target_node.node_type == NodeType.SOURCE:
                        edge_type = EdgeType.SELECTS_FROM
                    else:
                        edge_type = EdgeType.DEPENDS_ON
                    
                    edge = GraphEdge(
                        source_id=dependency_id,
                        target_id=node.unique_id,
                        edge_type=edge_type,
                        properties={
                            'relationship_source': 'manifest_dependencies'
                        }
                    )
                    self.edges.append(edge)
    
    def _create_column_relationships(self) -> None:
        """Create HAS_COLUMN relationships between nodes and their columns."""
        for node in self.nodes.values():
            if node.node_type == NodeType.COLUMN:
                parent_node_id = node.properties.get('parent_node')
                if parent_node_id and parent_node_id in self.nodes:
                    edge = GraphEdge(
                        source_id=parent_node_id,
                        target_id=node.unique_id,
                        edge_type=EdgeType.HAS_COLUMN,
                        properties={
                            'data_type': node.properties.get('data_type'),
                            'relationship_source': 'manifest_columns'
                        }
                    )
                    self.edges.append(edge)
    
    def _create_package_relationships(self) -> None:
        """Create PART_OF_PACKAGE relationships."""
        for node in self.nodes.values():
            package_name = node.properties.get('package_name')
            if package_name and node.node_type != NodeType.PACKAGE:
                package_id = f"package.{package_name}"
                if package_id in self.nodes:
                    edge = GraphEdge(
                        source_id=package_id,
                        target_id=node.unique_id,
                        edge_type=EdgeType.PART_OF_PACKAGE,
                        properties={
                            'relationship_source': 'manifest_packages'
                        }
                    )
                    self.edges.append(edge)
    
    def _create_test_relationships(self) -> None:
        """Create TESTS relationships from test nodes to what they test."""
        for node in self.nodes.values():
            if node.node_type == NodeType.TEST:
                depends_on_nodes = node.properties.get('depends_on_nodes', [])
                
                for tested_node_id in depends_on_nodes:
                    if tested_node_id in self.nodes:
                        edge = GraphEdge(
                            source_id=node.unique_id,
                            target_id=tested_node_id,
                            edge_type=EdgeType.TESTS,
                            properties={
                                'test_metadata': node.properties.get('test_metadata', {}),
                                'relationship_source': 'manifest_tests'
                            }
                        )
                        self.edges.append(edge)
    
    def _create_exposure_relationships(self) -> None:
        """Create USES relationships from exposures to models/metrics."""
        for node in self.nodes.values():
            if node.node_type == NodeType.EXPOSURE:
                depends_on_nodes = node.properties.get('depends_on_nodes', [])
                
                for dependency_id in depends_on_nodes:
                    if dependency_id in self.nodes:
                        edge = GraphEdge(
                            source_id=node.unique_id,
                            target_id=dependency_id,
                            edge_type=EdgeType.USES,
                            properties={
                                'exposure_type': node.properties.get('type'),
                                'relationship_source': 'manifest_exposures'
                            }
                        )
                        self.edges.append(edge)
    
    def _create_metric_relationships(self) -> None:
        """Create MEASURES relationships from metrics to models."""
        for node in self.nodes.values():
            if node.node_type == NodeType.METRIC:
                depends_on_nodes = node.properties.get('depends_on_nodes', [])
                
                for dependency_id in depends_on_nodes:
                    if dependency_id in self.nodes:
                        edge = GraphEdge(
                            source_id=node.unique_id,
                            target_id=dependency_id,
                            edge_type=EdgeType.MEASURES,
                            properties={
                                'metric_type': node.properties.get('type'),
                                'relationship_source': 'manifest_metrics'
                            }
                        )
                        self.edges.append(edge)
    
    def create_networkx_graph(self) -> nx.MultiDiGraph:
        """Create a NetworkX graph from the parsed nodes and edges."""
        self.logger.info("Creating NetworkX graph...")
        
        # Add nodes
        for unique_id, node in self.nodes.items():
            self.graph.add_node(unique_id, **node.to_dict())
        
        # Add edges
        for edge in self.edges:
            self.graph.add_edge(
                edge.source_id,
                edge.target_id,
                **edge.to_dict()
            )
        
        self.logger.info(f"Created NetworkX graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def build_and_store_graph(self) -> nx.MultiDiGraph:
        """
        Parse manifest, build graph, and store all outputs.
        This is the main workflow method that replaces the need for separate scripts.
        
        Returns:
            NetworkX MultiDiGraph object
        """
        self.logger.info(f"Building knowledge graph from manifest: {self.manifest_path}")
        
        # Parse manifest
        nodes, edges = self.parse_manifest()
        
        # Create NetworkX graph
        graph = self.create_networkx_graph()
        
        # Store all graph data
        self._save_graph_data(graph, nodes, edges, self.get_statistics())
        
        self.logger.info(f"Knowledge graph built and stored with {len(nodes)} nodes and {len(edges)} edges")
        return graph
    
    def load_graph(self) -> Optional[nx.MultiDiGraph]:
        """
        Load graph from stored files.
        
        Returns:
            NetworkX MultiDiGraph object or None if loading fails
        """
        try:
            # Try to load from pickle first (fastest)
            if self.graph_file.exists():
                self.logger.info(f"Loading graph from pickle: {self.graph_file}")
                with open(self.graph_file, 'rb') as f:
                    graph = pickle.load(f)
                self.logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
                return graph
            
            # Fall back to JSON files
            elif self.nodes_file.exists() and self.edges_file.exists():
                self.logger.info("Loading graph from JSON files")
                return self._load_graph_from_json()
            
            else:
                self.logger.warning("No stored graph files found")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load graph: {e}")
            return None
    
    def _load_graph_from_json(self) -> nx.MultiDiGraph:
        """Load graph from JSON node and edge files."""
        graph = nx.MultiDiGraph()
        
        # Load nodes
        with open(self.nodes_file, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)
        
        for node_data in nodes_data.copy():
            node_id = node_data.pop('id')
            graph.add_node(node_id, **node_data)
        
        # Load edges
        with open(self.edges_file, 'r', encoding='utf-8') as f:
            edges_data = json.load(f)
        
        for edge_data in edges_data.copy():
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            graph.add_edge(source, target, **edge_data)
        
        return graph
    
    def _save_graph_data(self, graph: nx.MultiDiGraph, nodes: Dict[str, GraphNode], 
                        edges: List[GraphEdge], statistics: Dict[str, Any]) -> None:
        """Save graph data to various formats."""
        
        # Save as pickle (for fast loading)
        self.logger.info(f"Saving graph to pickle: {self.graph_file}")
        with open(self.graph_file, 'wb') as f:
            pickle.dump(graph, f)
        
        # Save nodes as JSON
        self.logger.info(f"Saving nodes to JSON: {self.nodes_file}")
        nodes_data = [node.to_dict() for node in nodes.values()]
        with open(self.nodes_file, 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, indent=2, default=str)
        
        # Save edges as JSON
        self.logger.info(f"Saving edges to JSON: {self.edges_file}")
        edges_data = [edge.to_dict() for edge in edges]
        with open(self.edges_file, 'w', encoding='utf-8') as f:
            json.dump(edges_data, f, indent=2, default=str)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'statistics': statistics,
            'file_info': {
                'nodes_file': str(self.nodes_file),
                'edges_file': str(self.edges_file),
                'graph_file': str(self.graph_file)
            }
        }
        
        self.logger.info(f"Saving metadata to: {self.metadata_file}")
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def export_to_json(self, nodes_file: str, edges_file: str) -> None:
        """Export nodes and edges to specific JSON files (legacy method)."""
        self.logger.info(f"Exporting nodes to {nodes_file} and edges to {edges_file}")
        
        # Export nodes
        nodes_data = [node.to_dict() for node in self.nodes.values()]
        with open(nodes_file, 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, indent=2, default=str)
        
        # Export edges
        edges_data = [edge.to_dict() for edge in self.edges]
        with open(edges_file, 'w', encoding='utf-8') as f:
            json.dump(edges_data, f, indent=2, default=str)
        
        self.logger.info("Export completed successfully")
    
    def export_for_visualization(self, output_dir: str = "visualization") -> Dict[str, str]:
        """
        Export graph data in formats suitable for visualization tools.
        
        Args:
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary with file paths of exported files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        graph = self.load_graph()
        if not graph:
            # If no stored graph, create it from current state
            if not self.nodes:
                self.logger.warning("No graph data available. Parse manifest first.")
                return {}
            graph = self.create_networkx_graph()
        
        exported_files = {}
        
        # Export as GraphML (for Gephi, Cytoscape)
        # Need to convert complex data types to strings for GraphML compatibility
        graphml_graph = graph.copy()
        for node_id, attrs in graphml_graph.nodes(data=True):
            for key, value in attrs.items():
                if isinstance(value, (list, dict)):
                    attrs[key] = json.dumps(value) if value else ""
                elif value is None:
                    attrs[key] = ""
                elif not isinstance(value, (str, int, float, bool)):
                    attrs[key] = str(value)
        
        for u, v, attrs in graphml_graph.edges(data=True):
            for key, value in attrs.items():
                if isinstance(value, (list, dict)):
                    attrs[key] = json.dumps(value) if value else ""
                elif value is None:
                    attrs[key] = ""
                elif not isinstance(value, (str, int, float, bool)):
                    attrs[key] = str(value)
        
        graphml_file = output_path / "knowledge_graph.graphml"
        self.logger.info(f"Exporting to GraphML: {graphml_file}")
        nx.write_graphml(graphml_graph, graphml_file)
        exported_files['graphml'] = str(graphml_file)
        
        # Export as GML (for various tools)
        # GML also has issues with complex data types, so use the cleaned graph
        gml_file = output_path / "knowledge_graph.gml"
        self.logger.info(f"Exporting to GML: {gml_file}")
        try:
            nx.write_gml(graphml_graph, gml_file)  # Use the cleaned graph
            exported_files['gml'] = str(gml_file)
        except Exception as e:
            self.logger.warning(f"GML export failed: {e}. Skipping GML export.")
            exported_files['gml'] = "failed"
        
        # Export node and edge lists for web visualization
        vis_nodes = []
        vis_edges = []
        
        for node_id, attrs in graph.nodes(data=True):
            vis_node = {
                'id': node_id,
                'label': attrs.get('name', node_id),
                'type': attrs.get('node_type', 'unknown'),
                'group': attrs.get('node_type', 'unknown')
            }
            # Add important properties
            for key in ['description', 'package_name', 'resource_type']:
                if key in attrs:
                    vis_node[key] = attrs[key]
            vis_nodes.append(vis_node)
        
        for source, target, attrs in graph.edges(data=True):
            vis_edge = {
                'source': source,
                'target': target,
                'type': attrs.get('edge_type', 'related'),
                'label': attrs.get('edge_type', 'related')
            }
            vis_edges.append(vis_edge)
        
        # Save visualization data
        vis_data = {
            'nodes': vis_nodes,
            'edges': vis_edges,
            'metadata': self.get_stored_metadata()
        }
        
        vis_file = output_path / "visualization_data.json"
        self.logger.info(f"Exporting visualization data: {vis_file}")
        with open(vis_file, 'w', encoding='utf-8') as f:
            json.dump(vis_data, f, indent=2, default=str)
        exported_files['visualization'] = str(vis_file)
        
        return exported_files
    
    def get_stored_metadata(self) -> Optional[Dict[str, Any]]:
        """Get stored graph metadata."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        node_counts = {}
        for node_type in NodeType:
            count = len([n for n in self.nodes.values() if n.node_type == node_type])
            node_counts[node_type.value] = count
        
        edge_counts = {}
        for edge_type in EdgeType:
            count = len([e for e in self.edges if e.edge_type == edge_type])
            edge_counts[edge_type.value] = count
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_counts': node_counts,
            'edge_counts': edge_counts,
            'manifest_version': self.manifest_data.get('metadata', {}).get('dbt_version', 'unknown')
        }


def main():
    """Main function for the consolidated manifest parser and knowledge graph builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build knowledge graph from dbt manifest.json file')
    parser.add_argument('manifest_path', help='Path to manifest.json file')
    parser.add_argument('--storage-dir', default='data', help='Storage directory for graph files')
    parser.add_argument('--export-viz', action='store_true', help='Export visualization files')
    parser.add_argument('--load-only', action='store_true', help='Load existing graph instead of rebuilding')
    parser.add_argument('--legacy-export', help='Export to specific files (format: nodes.json,edges.json)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize parser
    manifest_parser = ManifestParser(args.manifest_path, args.storage_dir)
    
    if args.load_only:
        # Load existing graph
        graph = manifest_parser.load_graph()
        if not graph:
            print("No existing graph found. Use without --load-only to build new graph.")
            return
        stats = manifest_parser.get_stored_metadata()
        if stats and 'statistics' in stats:
            stats = stats['statistics']
        else:
            stats = {'total_nodes': graph.number_of_nodes(), 'total_edges': graph.number_of_edges()}
    else:
        # Build and store graph (main workflow)
        graph = manifest_parser.build_and_store_graph()
        stats = manifest_parser.get_statistics()
    
    # Legacy export if requested
    if args.legacy_export:
        files = args.legacy_export.split(',')
        if len(files) == 2:
            manifest_parser.export_to_json(files[0].strip(), files[1].strip())
            print(f"\nLegacy export completed: {files[0]}, {files[1]}")
    
    # Export visualization files if requested
    if args.export_viz:
        print("\nExporting visualization files...")
        exported = manifest_parser.export_for_visualization()
        for format_name, file_path in exported.items():
            print(f"  {format_name}: {file_path}")
    
    # Print statistics
    print("\nKnowledge Graph Statistics:")
    print(f"Total nodes: {stats.get('total_nodes', 'unknown')}")
    print(f"Total edges: {stats.get('total_edges', 'unknown')}")
    if 'manifest_version' in stats:
        print(f"Manifest version: {stats['manifest_version']}")
    
    if 'node_counts' in stats:
        print("\nNode counts by type:")
        for node_type, count in stats['node_counts'].items():
            if count > 0:
                print(f"  {node_type}: {count}")
    
    if 'edge_counts' in stats:
        print("\nEdge counts by type:")
        for edge_type, count in stats['edge_counts'].items():
            if count > 0:
                print(f"  {edge_type}: {count}")
    
    print(f"\nOutput files stored in: {manifest_parser.storage_dir}")


if __name__ == "__main__":
    main()