"""
Graph analysis and query module for the knowledge graph.
"""
import networkx as nx
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph_entities import NodeType, EdgeType


class GraphAnalyzer:
    """Analyze and query the dbt knowledge graph."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        """Initialize the graph analyzer."""
        self.graph = graph
    
    def get_model_lineage(self, model_id: str, direction: str = 'both', max_depth: int = None) -> Dict[str, List[str]]:
        """Get lineage for a specific model."""
        if model_id not in self.graph:
            return {'upstream': [], 'downstream': []}
        
        upstream = []
        downstream = []
        
        if direction in ['upstream', 'both']:
            if max_depth is None:
                upstream = list(nx.ancestors(self.graph, model_id))
            else:
                upstream = self._get_neighbors_by_depth(model_id, 'predecessors', max_depth)
        
        if direction in ['downstream', 'both']:
            if max_depth is None:
                downstream = list(nx.descendants(self.graph, model_id))
            else:
                downstream = self._get_neighbors_by_depth(model_id, 'successors', max_depth)
        
        return {
            'upstream': [node for node in upstream if self._get_node_type(node) == 'model'],
            'downstream': [node for node in downstream if self._get_node_type(node) == 'model']
        }
    
    def get_impact_analysis(self, node_id: str) -> Dict[str, Any]:
        """Analyze the impact of changes to a specific node."""
        if node_id not in self.graph:
            return {'error': f'Node {node_id} not found'}
        
        # Get all downstream nodes
        affected_nodes = list(nx.descendants(self.graph, node_id))
        
        # Categorize by node type
        impact_by_type = defaultdict(list)
        for node in affected_nodes:
            node_type = self._get_node_type(node)
            impact_by_type[node_type].append({
                'id': node,
                'name': self.graph.nodes[node].get('name', 'Unknown'),
                'distance': nx.shortest_path_length(self.graph, node_id, node)
            })
        
        # Sort by distance
        for node_type in impact_by_type:
            impact_by_type[node_type].sort(key=lambda x: x['distance'])
        
        return {
            'total_affected': len(affected_nodes),
            'by_type': dict(impact_by_type),
            'critical_paths': self._find_critical_paths(node_id)
        }
    
    def get_test_coverage(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze test coverage for models."""
        if model_id:
            if model_id not in self.graph:
                return {'error': f'Model {model_id} not found'}
            models = [model_id]
        else:
            models = [node for node in self.graph.nodes() if self._get_node_type(node) == 'model']
        
        coverage_data = []
        
        for model in models:
            # Find tests that validate this model
            tests = []
            for successor in self.graph.successors(model):
                if self._get_node_type(successor) == 'test':
                    edge_data = self.graph.get_edge_data(successor, model)
                    if edge_data and any(e.get('edge_type') == 'tests' for e in edge_data.values()):
                        tests.append({
                            'id': successor,
                            'name': self.graph.nodes[successor].get('name', 'Unknown'),
                            'test_type': self.graph.nodes[successor].get('test_type', 'Unknown')
                        })
            
            coverage_data.append({
                'model_id': model,
                'model_name': self.graph.nodes[model].get('name', 'Unknown'),
                'test_count': len(tests),
                'tests': tests,
                'has_coverage': len(tests) > 0
            })
        
        total_models = len(coverage_data)
        covered_models = sum(1 for m in coverage_data if m['has_coverage'])
        
        return {
            'total_models': total_models,
            'covered_models': covered_models,
            'coverage_percentage': (covered_models / total_models * 100) if total_models > 0 else 0,
            'details': coverage_data
        }
    
    def get_performance_analysis(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Analyze model and test performance."""
        performance_data = {
            'model_performance': [],
            'test_performance': [],
            'summary': {}
        }
        
        # Analyze model performance
        for node_id in self.graph.nodes():
            if self._get_node_type(node_id) == 'model':
                # Get execution edges for this model
                execution_times = []
                for pred in self.graph.predecessors(node_id):
                    if self._get_node_type(pred) == 'invocation':
                        edge_data = self.graph.get_edge_data(pred, node_id)
                        if edge_data:
                            for edge in edge_data.values():
                                if edge.get('edge_type') == 'executed_in':
                                    exec_time = edge.get('execution_time')
                                    if exec_time is not None:
                                        execution_times.append(float(exec_time))
                
                if execution_times:
                    performance_data['model_performance'].append({
                        'model_id': node_id,
                        'model_name': self.graph.nodes[node_id].get('name', 'Unknown'),
                        'avg_execution_time': sum(execution_times) / len(execution_times),
                        'max_execution_time': max(execution_times),
                        'min_execution_time': min(execution_times),
                        'execution_count': len(execution_times)
                    })
        
        # Sort by average execution time
        performance_data['model_performance'].sort(key=lambda x: x['avg_execution_time'], reverse=True)
        
        # Analyze test performance
        for node_id in self.graph.nodes():
            if self._get_node_type(node_id) == 'test':
                execution_times = []
                failure_count = 0
                total_runs = 0
                
                for pred in self.graph.predecessors(node_id):
                    if self._get_node_type(pred) == 'invocation':
                        edge_data = self.graph.get_edge_data(pred, node_id)
                        if edge_data:
                            for edge in edge_data.values():
                                if edge.get('edge_type') == 'has_result':
                                    total_runs += 1
                                    exec_time = edge.get('execution_time')
                                    if exec_time is not None:
                                        execution_times.append(float(exec_time))
                                    
                                    status = edge.get('status')
                                    if status in ['fail', 'error']:
                                        failure_count += 1
                
                if total_runs > 0:
                    performance_data['test_performance'].append({
                        'test_id': node_id,
                        'test_name': self.graph.nodes[node_id].get('name', 'Unknown'),
                        'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
                        'failure_rate': (failure_count / total_runs) * 100,
                        'total_runs': total_runs,
                        'failure_count': failure_count
                    })
        
        # Sort by failure rate
        performance_data['test_performance'].sort(key=lambda x: x['failure_rate'], reverse=True)
        
        # Summary statistics
        if performance_data['model_performance']:
            model_times = [m['avg_execution_time'] for m in performance_data['model_performance']]
            performance_data['summary']['avg_model_time'] = sum(model_times) / len(model_times)
            performance_data['summary']['slowest_models'] = performance_data['model_performance'][:5]
        
        if performance_data['test_performance']:
            failure_rates = [t['failure_rate'] for t in performance_data['test_performance']]
            performance_data['summary']['avg_test_failure_rate'] = sum(failure_rates) / len(failure_rates)
            performance_data['summary']['most_failing_tests'] = performance_data['test_performance'][:5]
        
        return performance_data
    
    def find_data_quality_issues(self) -> Dict[str, Any]:
        """Identify potential data quality issues from test results."""
        issues = {
            'failing_tests': [],
            'models_without_tests': [],
            'frequently_failing_tests': [],
            'summary': {}
        }
        
        # Find currently failing tests
        for node_id in self.graph.nodes():
            if self._get_node_type(node_id) == 'test':
                # Get latest test result
                latest_result = None
                latest_time = None
                
                for pred in self.graph.predecessors(node_id):
                    if self._get_node_type(pred) == 'invocation':
                        edge_data = self.graph.get_edge_data(pred, node_id)
                        if edge_data:
                            for edge in edge_data.values():
                                if edge.get('edge_type') == 'has_result':
                                    detected_at = edge.get('detected_at')
                                    if latest_time is None or (detected_at and detected_at > latest_time):
                                        latest_time = detected_at
                                        latest_result = edge
                
                if latest_result and latest_result.get('status') in ['fail', 'error', 'warn']:
                    issues['failing_tests'].append({
                        'test_id': node_id,
                        'test_name': self.graph.nodes[node_id].get('name', 'Unknown'),
                        'status': latest_result.get('status'),
                        'result_rows': latest_result.get('result_rows'),
                        'detected_at': latest_result.get('detected_at'),
                        'message': latest_result.get('test_message')
                    })
        
        # Find models without tests
        test_coverage = self.get_test_coverage()
        issues['models_without_tests'] = [
            model for model in test_coverage['details'] 
            if not model['has_coverage']
        ]
        
        # Summary
        issues['summary'] = {
            'total_failing_tests': len(issues['failing_tests']),
            'models_without_tests_count': len(issues['models_without_tests']),
            'test_coverage_percentage': test_coverage['coverage_percentage']
        }
        
        return issues
    
    def get_dependency_complexity(self) -> Dict[str, Any]:
        """Analyze dependency complexity of the graph."""
        complexity_metrics = {
            'models_by_dependency_count': [],
            'circular_dependencies': [],
            'longest_paths': [],
            'summary': {}
        }
        
        # Analyze dependency counts
        for node_id in self.graph.nodes():
            if self._get_node_type(node_id) == 'model':
                predecessors = list(self.graph.predecessors(node_id))
                successors = list(self.graph.successors(node_id))
                
                # Count only model dependencies
                model_predecessors = [p for p in predecessors if self._get_node_type(p) == 'model']
                model_successors = [s for s in successors if self._get_node_type(s) == 'model']
                
                complexity_metrics['models_by_dependency_count'].append({
                    'model_id': node_id,
                    'model_name': self.graph.nodes[node_id].get('name', 'Unknown'),
                    'upstream_count': len(model_predecessors),
                    'downstream_count': len(model_successors),
                    'total_dependencies': len(model_predecessors) + len(model_successors)
                })
        
        # Sort by total dependencies
        complexity_metrics['models_by_dependency_count'].sort(
            key=lambda x: x['total_dependencies'], reverse=True
        )
        
        # Check for circular dependencies
        try:
            cycles = list(nx.simple_cycles(self.graph))
            complexity_metrics['circular_dependencies'] = [
                {'cycle': cycle, 'length': len(cycle)} 
                for cycle in cycles[:10]  # Limit to first 10
            ]
        except:
            complexity_metrics['circular_dependencies'] = []
        
        # Find longest paths
        model_nodes = [n for n in self.graph.nodes() if self._get_node_type(n) == 'model']
        if model_nodes:
            try:
                # Find some long paths (this can be expensive for large graphs)
                longest_paths = []
                for source in model_nodes[:10]:  # Limit source nodes
                    for target in model_nodes[:10]:  # Limit target nodes
                        if source != target:
                            try:
                                path = nx.shortest_path(self.graph, source, target)
                                longest_paths.append({
                                    'source': source,
                                    'target': target,
                                    'path': path,
                                    'length': len(path)
                                })
                            except nx.NetworkXNoPath:
                                continue
                
                longest_paths.sort(key=lambda x: x['length'], reverse=True)
                complexity_metrics['longest_paths'] = longest_paths[:5]
            except:
                complexity_metrics['longest_paths'] = []
        
        # Summary
        if complexity_metrics['models_by_dependency_count']:
            dependency_counts = [m['total_dependencies'] for m in complexity_metrics['models_by_dependency_count']]
            complexity_metrics['summary'] = {
                'total_models': len(complexity_metrics['models_by_dependency_count']),
                'avg_dependencies_per_model': sum(dependency_counts) / len(dependency_counts),
                'max_dependencies': max(dependency_counts),
                'models_with_high_complexity': len([m for m in complexity_metrics['models_by_dependency_count'] if m['total_dependencies'] > 10]),
                'circular_dependencies_count': len(complexity_metrics['circular_dependencies'])
            }
        
        return complexity_metrics
    
    def _get_node_type(self, node_id: str) -> str:
        """Get the type of a node."""
        return self.graph.nodes[node_id].get('node_type', 'unknown')
    
    def _get_neighbors_by_depth(self, node_id: str, direction: str, max_depth: int) -> List[str]:
        """Get neighbors up to a certain depth."""
        neighbors = set()
        current_level = {node_id}
        
        for _ in range(max_depth):
            next_level = set()
            for node in current_level:
                if direction == 'predecessors':
                    new_neighbors = set(self.graph.predecessors(node))
                else:  # successors
                    new_neighbors = set(self.graph.successors(node))
                
                next_level.update(new_neighbors)
                neighbors.update(new_neighbors)
            
            current_level = next_level
            if not current_level:
                break
        
        return list(neighbors)
    
    def _find_critical_paths(self, node_id: str) -> List[Dict[str, Any]]:
        """Find critical paths from a node (paths through important models)."""
        critical_paths = []
        
        # Get all paths to leaf nodes (nodes with no successors)
        descendants = nx.descendants(self.graph, node_id)
        leaf_nodes = [node for node in descendants if len(list(self.graph.successors(node))) == 0]
        
        for leaf in leaf_nodes[:5]:  # Limit to 5 paths
            try:
                path = nx.shortest_path(self.graph, node_id, leaf)
                # Filter to only model nodes
                model_path = [node for node in path if self._get_node_type(node) == 'model']
                
                if len(model_path) > 1:
                    critical_paths.append({
                        'path': model_path,
                        'length': len(model_path),
                        'end_node': leaf
                    })
            except nx.NetworkXNoPath:
                continue
        
        return sorted(critical_paths, key=lambda x: x['length'], reverse=True)