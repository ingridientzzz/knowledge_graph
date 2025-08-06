"""
Graph entity definitions and enums for the knowledge graph.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json


class NodeType(Enum):
    """Enumeration of node types in the knowledge graph."""
    MODEL = "model"
    TEST = "test"
    SOURCE = "source"
    COLUMN = "column"
    INVOCATION = "invocation"
    SNAPSHOT = "snapshot"


class EdgeType(Enum):
    """Enumeration of edge types in the knowledge graph."""
    DEPENDS_ON = "depends_on"
    HAS_COLUMN = "has_column"
    TESTS = "tests"
    FEEDS_INTO = "feeds_into"
    EXECUTED_IN = "executed_in"
    HAS_RESULT = "has_result"


class Status(Enum):
    """Enumeration of execution statuses."""
    SUCCESS = "success"
    FAIL = "fail" 
    ERROR = "error"
    WARN = "warn"
    PASS = "pass"
    SKIPPED = "skipped"


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    node_id: str
    node_type: NodeType
    name: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            'id': self.node_id,
            'type': self.node_type.value,
            'name': self.name,
            'properties': self.properties
        }


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            'source': self.source_id,
            'target': self.target_id,
            'type': self.edge_type.value,
            'properties': self.properties
        }


class NodeFactory:
    """Factory class for creating graph nodes from Elementary data."""
    
    @staticmethod
    def create_model_node(model_data: Dict[str, Any]) -> GraphNode:
        """Create a model node from dbt model data."""
        properties = {
            'database_name': model_data.get('database_name'),
            'schema_name': model_data.get('schema_name'),
            'table_name': model_data.get('table_name'),
            'materialization': model_data.get('materialization'),
            'package_name': model_data.get('package_name'),
            'path': model_data.get('path'),
            'alias': model_data.get('alias'),
            'tags': NodeFactory._parse_json_field(model_data.get('tags')),
            'meta': NodeFactory._parse_json_field(model_data.get('meta')),
            'generated_at': str(model_data.get('generated_at')),
            'full_name': f"{model_data.get('database_name', '')}.{model_data.get('schema_name', '')}.{model_data.get('table_name', '')}"
        }
        
        return GraphNode(
            node_id=model_data['unique_id'],
            node_type=NodeType.MODEL,
            name=model_data['name'],
            properties=properties
        )
    
    @staticmethod
    def create_test_node(test_data: Dict[str, Any]) -> GraphNode:
        """Create a test node from dbt test data."""
        properties = {
            'test_type': test_data.get('test_type'),
            'test_params': NodeFactory._parse_json_field(test_data.get('test_params')),
            'package_name': test_data.get('package_name'),
            'path': test_data.get('path'),
            'tags': NodeFactory._parse_json_field(test_data.get('tags')),
            'meta': NodeFactory._parse_json_field(test_data.get('meta')),
            'generated_at': str(test_data.get('generated_at'))
        }
        
        return GraphNode(
            node_id=test_data['unique_id'],
            node_type=NodeType.TEST,
            name=test_data['name'],
            properties=properties
        )
    
    @staticmethod
    def create_source_node(source_data: Dict[str, Any]) -> GraphNode:
        """Create a source node from dbt source data."""
        properties = {
            'source_name': source_data.get('source_name'),
            'database_name': source_data.get('database_name'),
            'schema_name': source_data.get('schema_name'),
            'table_name': source_data.get('table_name'),
            'tags': NodeFactory._parse_json_field(source_data.get('tags')),
            'meta': NodeFactory._parse_json_field(source_data.get('meta')),
            'generated_at': str(source_data.get('generated_at')),
            'full_name': f"{source_data.get('database_name', '')}.{source_data.get('schema_name', '')}.{source_data.get('table_name', '')}"
        }
        
        return GraphNode(
            node_id=source_data['unique_id'],
            node_type=NodeType.SOURCE,
            name=source_data['name'],
            properties=properties
        )
    
    @staticmethod
    def create_column_node(column_data: Dict[str, Any], model_id: str) -> GraphNode:
        """Create a column node from model column data."""
        column_id = f"{model_id}::{column_data['column_name']}"
        properties = {
            'data_type': column_data.get('data_type'),
            'is_nullable': column_data.get('is_nullable'),
            'column_index': column_data.get('column_index'),
            'model_id': model_id,
            'generated_at': str(column_data.get('generated_at'))
        }
        
        return GraphNode(
            node_id=column_id,
            node_type=NodeType.COLUMN,
            name=column_data['column_name'],
            properties=properties
        )
    
    @staticmethod
    def create_invocation_node(invocation_data: Dict[str, Any]) -> GraphNode:
        """Create an invocation node from dbt invocation data."""
        properties = {
            'job_name': invocation_data.get('job_name'),
            'command': invocation_data.get('command'),
            'dbt_version': invocation_data.get('dbt_version'),
            'is_full_refresh': invocation_data.get('is_full_refresh'),
            'env_vars': NodeFactory._parse_json_field(invocation_data.get('env_vars')),
            'invocation_time': str(invocation_data.get('invocation_time')),
            'generated_at': str(invocation_data.get('generated_at'))
        }
        
        return GraphNode(
            node_id=invocation_data['invocation_id'],
            node_type=NodeType.INVOCATION,
            name=f"Invocation {invocation_data['invocation_id']}",
            properties=properties
        )
    
    @staticmethod
    def _parse_json_field(field_value: Any) -> Any:
        """Parse JSON field value, handling both string and already parsed objects."""
        if isinstance(field_value, str):
            try:
                return json.loads(field_value)
            except (json.JSONDecodeError, TypeError):
                return field_value
        return field_value


class EdgeFactory:
    """Factory class for creating graph edges from Elementary data."""
    
    @staticmethod
    def create_dependency_edges(model_data: Dict[str, Any]) -> List[GraphEdge]:
        """Create dependency edges from model's depends_on_nodes."""
        edges = []
        depends_on = EdgeFactory._parse_depends_on(model_data.get('depends_on_nodes'))
        
        for dependency in depends_on:
            edge = GraphEdge(
                source_id=dependency,
                target_id=model_data['unique_id'],
                edge_type=EdgeType.DEPENDS_ON,
                properties={
                    'created_at': str(model_data.get('generated_at')),
                    'relationship_source': 'manifest'
                }
            )
            edges.append(edge)
        
        return edges
    
    @staticmethod
    def create_test_edges(test_data: Dict[str, Any]) -> List[GraphEdge]:
        """Create test edges from test's depends_on_nodes."""
        edges = []
        depends_on = EdgeFactory._parse_depends_on(test_data.get('depends_on_nodes'))
        
        for tested_node in depends_on:
            edge = GraphEdge(
                source_id=test_data['unique_id'],
                target_id=tested_node,
                edge_type=EdgeType.TESTS,
                properties={
                    'test_type': test_data.get('test_type'),
                    'created_at': str(test_data.get('generated_at')),
                    'relationship_source': 'manifest'
                }
            )
            edges.append(edge)
        
        return edges
    
    @staticmethod
    def create_column_edges(column_data: Dict[str, Any], model_id: str) -> GraphEdge:
        """Create edge between model and column."""
        column_id = f"{model_id}::{column_data['column_name']}"
        
        return GraphEdge(
            source_id=model_id,
            target_id=column_id,
            edge_type=EdgeType.HAS_COLUMN,
            properties={
                'data_type': column_data.get('data_type'),
                'is_nullable': column_data.get('is_nullable'),
                'column_index': column_data.get('column_index'),
                'created_at': str(column_data.get('generated_at'))
            }
        )
    
    @staticmethod
    def create_execution_edges(run_result: Dict[str, Any]) -> GraphEdge:
        """Create execution edge between invocation and model."""
        return GraphEdge(
            source_id=run_result['invocation_id'],
            target_id=run_result['model_unique_id'],
            edge_type=EdgeType.EXECUTED_IN,
            properties={
                'status': run_result.get('status'),
                'execution_time': run_result.get('execution_time'),
                'rows_affected': run_result.get('rows_affected'),
                'materialization': run_result.get('materialization'),
                'detected_at': str(run_result.get('detected_at')),
                'message': run_result.get('message')
            }
        )
    
    @staticmethod
    def create_test_result_edges(test_result: Dict[str, Any]) -> GraphEdge:
        """Create test result edge between invocation and test."""
        return GraphEdge(
            source_id=test_result['invocation_id'],
            target_id=test_result['test_unique_id'],
            edge_type=EdgeType.HAS_RESULT,
            properties={
                'status': test_result.get('status'),
                'result_rows': test_result.get('result_rows'),
                'execution_time': test_result.get('execution_time'),
                'detected_at': str(test_result.get('detected_at')),
                'test_message': test_result.get('test_message'),
                'compiled_sql': test_result.get('compiled_sql')
            }
        )
    
    @staticmethod
    def _parse_depends_on(depends_on_value: Any) -> List[str]:
        """Parse depends_on_nodes field value."""
        if isinstance(depends_on_value, str):
            try:
                parsed = json.loads(depends_on_value)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                return []
        elif isinstance(depends_on_value, list):
            return depends_on_value
        return []