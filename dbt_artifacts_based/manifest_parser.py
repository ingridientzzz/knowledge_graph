#!/usr/bin/env python3
"""
Comprehensive manifest.json parser for dbt projects.
Transforms dbt metadata into a graph-friendly format with nodes, relationships, and properties.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
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
    """Parser for dbt manifest.json files."""
    
    def __init__(self, manifest_path: str):
        """Initialize the parser with a manifest file path."""
        self.manifest_path = Path(manifest_path)
        self.manifest_data: Dict[str, Any] = {}
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.graph = nx.MultiDiGraph()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
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
        
        node = GraphNode(
            unique_id=unique_id,
            node_type=NodeType.SOURCE,
            name=source_data.get('name', ''),
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
    
    def export_to_json(self, nodes_file: str, edges_file: str) -> None:
        """Export nodes and edges to JSON files."""
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
    """Main function for testing the manifest parser."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse dbt manifest.json file')
    parser.add_argument('manifest_path', help='Path to manifest.json file')
    parser.add_argument('--output-dir', default='.', help='Output directory for JSON files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse manifest
    manifest_parser = ManifestParser(args.manifest_path)
    nodes, edges = manifest_parser.parse_manifest()
    
    # Export to JSON
    nodes_file = Path(args.output_dir) / 'nodes.json'
    edges_file = Path(args.output_dir) / 'edges.json'
    manifest_parser.export_to_json(str(nodes_file), str(edges_file))
    
    # Print statistics
    stats = manifest_parser.get_statistics()
    print("\nParsing Statistics:")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Manifest version: {stats['manifest_version']}")
    print("\nNode counts by type:")
    for node_type, count in stats['node_counts'].items():
        if count > 0:
            print(f"  {node_type}: {count}")
    print("\nEdge counts by type:")
    for edge_type, count in stats['edge_counts'].items():
        if count > 0:
            print(f"  {edge_type}: {count}")


if __name__ == "__main__":
    main()