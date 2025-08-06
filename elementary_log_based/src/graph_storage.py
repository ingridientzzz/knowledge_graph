"""
Graph storage and persistence module supporting multiple backends.
"""
import json
import pickle
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import networkx as nx
from neo4j import GraphDatabase
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database_config import DatabaseConfig


class GraphStorageBackend(ABC):
    """Abstract base class for graph storage backends."""
    
    @abstractmethod
    def save_graph(self, graph: nx.MultiDiGraph, metadata: Dict[str, Any]) -> bool:
        """Save graph to storage backend."""
        pass
    
    @abstractmethod
    def load_graph(self) -> Optional[nx.MultiDiGraph]:
        """Load graph from storage backend."""
        pass
    
    @abstractmethod
    def query_nodes(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query nodes with filters."""
        pass
    
    @abstractmethod
    def query_paths(self, start_node: str, end_node: str, max_depth: int = 5) -> List[List[str]]:
        """Find paths between nodes."""
        pass


class FileSystemStorage(GraphStorageBackend):
    """File system storage backend for graphs."""
    
    def __init__(self, base_path: str = "/Users/marquein/knowledge_graph/data"):
        """Initialize file system storage."""
        self.base_path = base_path
        self.graph_file = f"{base_path}/knowledge_graph.gpickle"
        self.metadata_file = f"{base_path}/graph_metadata.json"
        self.nodes_file = f"{base_path}/nodes.json"
        self.edges_file = f"{base_path}/edges.json"
    
    def save_graph(self, graph: nx.MultiDiGraph, metadata: Dict[str, Any]) -> bool:
        """Save graph to file system."""
        try:
            # Save NetworkX graph
            import pickle
            with open(self.graph_file, 'wb') as f:
                pickle.dump(graph, f)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Export nodes and edges as JSON for easy querying
            nodes_data = []
            for node_id, node_attrs in graph.nodes(data=True):
                node_data = {'id': node_id, **node_attrs}
                nodes_data.append(node_data)
            
            with open(self.nodes_file, 'w') as f:
                json.dump(nodes_data, f, indent=2, default=str)
            
            edges_data = []
            for source, target, edge_attrs in graph.edges(data=True):
                edge_data = {'source': source, 'target': target, **edge_attrs}
                edges_data.append(edge_data)
            
            with open(self.edges_file, 'w') as f:
                json.dump(edges_data, f, indent=2, default=str)
            
            print(f"✅ Graph saved to {self.base_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save graph: {e}")
            return False
    
    def load_graph(self) -> Optional[nx.MultiDiGraph]:
        """Load graph from file system."""
        try:
            import pickle
            with open(self.graph_file, 'rb') as f:
                graph = pickle.load(f)
            print(f"✅ Graph loaded from {self.graph_file}")
            return graph
        except Exception as e:
            print(f"❌ Failed to load graph: {e}")
            return None
    
    def query_nodes(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query nodes with filters."""
        try:
            with open(self.nodes_file, 'r') as f:
                nodes = json.load(f)
            
            filtered_nodes = []
            for node in nodes:
                match = True
                for key, value in filters.items():
                    if key not in node or node[key] != value:
                        match = False
                        break
                if match:
                    filtered_nodes.append(node)
            
            return filtered_nodes
            
        except Exception as e:
            print(f"❌ Failed to query nodes: {e}")
            return []
    
    def query_paths(self, start_node: str, end_node: str, max_depth: int = 5) -> List[List[str]]:
        """Find paths between nodes."""
        graph = self.load_graph()
        if not graph:
            return []
        
        try:
            paths = list(nx.all_simple_paths(graph, start_node, end_node, cutoff=max_depth))
            return paths
        except Exception as e:
            print(f"❌ Failed to find paths: {e}")
            return []


class Neo4jStorage(GraphStorageBackend):
    """Neo4j storage backend for graphs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Neo4j storage."""
        self.config = config or DatabaseConfig.get_neo4j_config()
        self.driver = None
    
    def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.config['uri'],
                auth=(self.config['user'], self.config['password'])
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✅ Connected to Neo4j successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Neo4j."""
        if self.driver:
            self.driver.close()
            print("🔌 Disconnected from Neo4j")
    
    def save_graph(self, graph: nx.MultiDiGraph, metadata: Dict[str, Any]) -> bool:
        """Save graph to Neo4j."""
        if not self.driver and not self.connect():
            return False
        
        try:
            with self.driver.session() as session:
                # Clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                
                # Create nodes
                for node_id, node_attrs in graph.nodes(data=True):
                    node_type = node_attrs.get('node_type', 'Unknown')
                    properties = {k: v for k, v in node_attrs.items() if v is not None}
                    
                    # Convert lists and dicts to strings for Neo4j
                    for key, value in properties.items():
                        if isinstance(value, (list, dict)):
                            properties[key] = json.dumps(value)
                    
                    props_str = ', '.join(f'{k}: ${k}' for k in properties.keys())
                    session.run(
                        f"CREATE (n:{node_type} {{id: $id, {props_str}}})",
                        id=node_id, **properties
                    )
                
                # Create relationships
                for source, target, edge_attrs in graph.edges(data=True):
                    edge_type = edge_attrs.get('edge_type', 'RELATED_TO').upper()
                    properties = {k: v for k, v in edge_attrs.items() if v is not None and k != 'edge_type'}
                    
                    # Convert lists and dicts to strings
                    for key, value in properties.items():
                        if isinstance(value, (list, dict)):
                            properties[key] = json.dumps(value)
                    
                    edge_props_str = ', '.join(f'{k}: ${k}' for k in properties.keys())
                    session.run(
                        f"""
                        MATCH (a {{id: $source}})
                        MATCH (b {{id: $target}})
                        CREATE (a)-[r:{edge_type} {{{edge_props_str}}}]->(b)
                        """,
                        source=source, target=target, **properties
                    )
                
                print(f"✅ Graph saved to Neo4j with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
                return True
                
        except Exception as e:
            print(f"❌ Failed to save graph to Neo4j: {e}")
            return False
    
    def load_graph(self) -> Optional[nx.MultiDiGraph]:
        """Load graph from Neo4j."""
        if not self.driver and not self.connect():
            return None
        
        try:
            graph = nx.MultiDiGraph()
            
            with self.driver.session() as session:
                # Load nodes
                result = session.run("MATCH (n) RETURN n")
                for record in result:
                    node = record['n']
                    node_id = node['id']
                    properties = dict(node)
                    graph.add_node(node_id, **properties)
                
                # Load edges
                result = session.run("MATCH (a)-[r]->(b) RETURN a.id, b.id, type(r), properties(r)")
                for record in result:
                    source = record['a.id']
                    target = record['b.id']
                    edge_type = record['type(r)']
                    properties = dict(record['properties(r)'])
                    properties['edge_type'] = edge_type.lower()
                    graph.add_edge(source, target, **properties)
            
            print(f"✅ Graph loaded from Neo4j with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph
            
        except Exception as e:
            print(f"❌ Failed to load graph from Neo4j: {e}")
            return None
    
    def query_nodes(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query nodes with filters."""
        if not self.driver and not self.connect():
            return []
        
        try:
            with self.driver.session() as session:
                # Build WHERE clause
                where_conditions = []
                params = {}
                for key, value in filters.items():
                    where_conditions.append(f"n.{key} = ${key}")
                    params[key] = value
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "true"
                query = f"MATCH (n) WHERE {where_clause} RETURN n"
                
                result = session.run(query, **params)
                nodes = [dict(record['n']) for record in result]
                return nodes
                
        except Exception as e:
            print(f"❌ Failed to query nodes: {e}")
            return []
    
    def query_paths(self, start_node: str, end_node: str, max_depth: int = 5) -> List[List[str]]:
        """Find paths between nodes."""
        if not self.driver and not self.connect():
            return []
        
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH path = (start {{id: $start_node}})-[*1..{max_depth}]->(end {{id: $end_node}})
                RETURN [node in nodes(path) | node.id] as path_nodes
                LIMIT 100
                """
                
                result = session.run(query, start_node=start_node, end_node=end_node)
                paths = [record['path_nodes'] for record in result]
                return paths
                
        except Exception as e:
            print(f"❌ Failed to find paths: {e}")
            return []


class GraphStorageManager:
    """Manager for different graph storage backends."""
    
    def __init__(self, backend_type: str = 'filesystem', **kwargs):
        """Initialize storage manager with specified backend."""
        if backend_type == 'filesystem':
            self.backend = FileSystemStorage(**kwargs)
        elif backend_type == 'neo4j':
            self.backend = Neo4jStorage(**kwargs)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        
        self.backend_type = backend_type
    
    def save_graph(self, graph: nx.MultiDiGraph, metadata: Dict[str, Any] = None) -> bool:
        """Save graph using configured backend."""
        if metadata is None:
            metadata = {
                'nodes': len(graph.nodes),
                'edges': len(graph.edges),
                'backend': self.backend_type
            }
        
        return self.backend.save_graph(graph, metadata)
    
    def load_graph(self) -> Optional[nx.MultiDiGraph]:
        """Load graph using configured backend."""
        return self.backend.load_graph()
    
    def query_nodes(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query nodes with filters."""
        return self.backend.query_nodes(filters)
    
    def query_paths(self, start_node: str, end_node: str, max_depth: int = 5) -> List[List[str]]:
        """Find paths between nodes."""
        return self.backend.query_paths(start_node, end_node, max_depth)
    
    def export_to_format(self, graph: nx.MultiDiGraph, format_type: str, output_path: str) -> bool:
        """Export graph to various formats."""
        try:
            if format_type == 'gexf':
                nx.write_gexf(graph, output_path)
            elif format_type == 'graphml':
                nx.write_graphml(graph, output_path)
            elif format_type == 'json':
                graph_data = nx.node_link_data(graph)
                with open(output_path, 'w') as f:
                    json.dump(graph_data, f, indent=2, default=str)
            elif format_type == 'csv':
                # Export nodes and edges as separate CSV files
                nodes_df = pd.DataFrame([
                    {'id': node_id, **attrs} 
                    for node_id, attrs in graph.nodes(data=True)
                ])
                edges_df = pd.DataFrame([
                    {'source': source, 'target': target, **attrs}
                    for source, target, attrs in graph.edges(data=True)
                ])
                
                base_path = output_path.rsplit('.', 1)[0]
                nodes_df.to_csv(f"{base_path}_nodes.csv", index=False)
                edges_df.to_csv(f"{base_path}_edges.csv", index=False)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            print(f"✅ Graph exported to {format_type} format: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export graph: {e}")
            return False