#!/usr/bin/env python3
"""
Graph storage management for dbt artifacts-based knowledge graph.
Integrates with the manifest parser to store and retrieve graph data.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import networkx as nx
from datetime import datetime

from manifest_parser import ManifestParser, GraphNode, GraphEdge


class GraphStorageManager:
    """Manages storage and retrieval of graph data from dbt artifacts."""
    
    def __init__(self, storage_dir: str = "data"):
        """Initialize the storage manager."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # File paths
        self.nodes_file = self.storage_dir / "nodes.json"
        self.edges_file = self.storage_dir / "edges.json"
        self.graph_file = self.storage_dir / "knowledge_graph.gpickle"
        self.metadata_file = self.storage_dir / "graph_metadata.json"
    
    def build_graph_from_manifest(self, manifest_path: str) -> nx.MultiDiGraph:
        """
        Build and store graph from manifest.json file.
        
        Args:
            manifest_path: Path to the dbt manifest.json file
            
        Returns:
            NetworkX MultiDiGraph object
        """
        self.logger.info(f"Building graph from manifest: {manifest_path}")
        
        # Parse manifest
        parser = ManifestParser(manifest_path)
        nodes, edges = parser.parse_manifest()
        
        # Create NetworkX graph
        graph = parser.create_networkx_graph()
        
        # Store graph data
        self._save_graph_data(graph, nodes, edges, parser.get_statistics())
        
        self.logger.info(f"Graph built and stored with {len(nodes)} nodes and {len(edges)} edges")
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
        
        for node_data in nodes_data:
            node_id = node_data.pop('id')
            graph.add_node(node_id, **node_data)
        
        # Load edges
        with open(self.edges_file, 'r', encoding='utf-8') as f:
            edges_data = json.load(f)
        
        for edge_data in edges_data:
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
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get stored graph metadata."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None
    
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
            raise ValueError("No graph data available for export")
        
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
            'metadata': self.get_metadata()
        }
        
        vis_file = output_path / "visualization_data.json"
        self.logger.info(f"Exporting visualization data: {vis_file}")
        with open(vis_file, 'w', encoding='utf-8') as f:
            json.dump(vis_data, f, indent=2, default=str)
        exported_files['visualization'] = str(vis_file)
        
        return exported_files
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current graph statistics."""
        metadata = self.get_metadata()
        if metadata and 'statistics' in metadata:
            return metadata['statistics']
        
        # Calculate on-the-fly if no stored statistics
        graph = self.load_graph()
        if graph:
            return {
                'total_nodes': graph.number_of_nodes(),
                'total_edges': graph.number_of_edges(),
                'calculated_at': datetime.now().isoformat()
            }
        
        return {}
    
    def cleanup_old_files(self, keep_backups: int = 3) -> None:
        """Clean up old graph files, keeping the specified number of backups."""
        # This could be implemented to manage multiple versions of graph files
        # For now, it's a placeholder for future backup management
        pass


def main():
    """Main function for testing the graph storage manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build knowledge graph from dbt manifest')
    parser.add_argument('manifest_path', help='Path to manifest.json file')
    parser.add_argument('--storage-dir', default='data', help='Storage directory for graph files')
    parser.add_argument('--export-viz', action='store_true', help='Export visualization files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Build graph
    storage_manager = GraphStorageManager(args.storage_dir)
    graph = storage_manager.build_graph_from_manifest(args.manifest_path)
    
    # Print statistics
    stats = storage_manager.get_statistics()
    print("\nGraph Statistics:")
    if 'total_nodes' in stats:
        print(f"Total nodes: {stats['total_nodes']}")
    if 'total_edges' in stats:
        print(f"Total edges: {stats['total_edges']}")
    if 'node_counts' in stats:
        print("\nNode counts by type:")
        for node_type, count in stats['node_counts'].items():
            if count > 0:
                print(f"  {node_type}: {count}")
    
    # Export visualization files if requested
    if args.export_viz:
        print("\nExporting visualization files...")
        exported = storage_manager.export_for_visualization()
        for format_name, file_path in exported.items():
            print(f"  {format_name}: {file_path}")


if __name__ == "__main__":
    main()