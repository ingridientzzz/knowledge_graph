"""
Advanced query examples for the dbt Elementary Knowledge Graph.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.graph_storage import GraphStorageManager
from src.graph_analyzer import GraphAnalyzer
import networkx as nx
from typing import List, Dict, Any


class AdvancedGraphQueries:
    """Advanced queries for the dbt knowledge graph."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        """Initialize with a graph."""
        self.graph = graph
        self.analyzer = GraphAnalyzer(graph)
    
    def find_models_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Find all models with a specific tag."""
        models = []
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if node_data.get('node_type') == 'model':
                tags = node_data.get('tags', [])
                if isinstance(tags, list) and tag in tags:
                    models.append({
                        'id': node_id,
                        'name': node_data.get('name'),
                        'schema': node_data.get('schema_name'),
                        'materialization': node_data.get('materialization')
                    })
        return models
    
    def find_models_in_schema(self, schema_name: str) -> List[Dict[str, Any]]:
        """Find all models in a specific schema."""
        models = []
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if (node_data.get('node_type') == 'model' and 
                node_data.get('schema_name') == schema_name):
                models.append({
                    'id': node_id,
                    'name': node_data.get('name'),
                    'table_name': node_data.get('table_name'),
                    'materialization': node_data.get('materialization')
                })
        return models
    
    def find_longest_dependency_chains(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Find the longest dependency chains in the graph."""
        chains = []
        model_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('node_type') == 'model']
        
        # Find paths between all model pairs
        for source in model_nodes:
            for target in model_nodes:
                if source != target:
                    try:
                        path = nx.shortest_path(self.graph, source, target)
                        # Filter to only model nodes in the path
                        model_path = [n for n in path if self.graph.nodes[n].get('node_type') == 'model']
                        
                        if len(model_path) > 2:  # More than just source and target
                            chains.append({
                                'source': source,
                                'target': target,
                                'path': model_path,
                                'length': len(model_path),
                                'model_names': [self.graph.nodes[n].get('name', 'Unknown') for n in model_path]
                            })
                    except nx.NetworkXNoPath:
                        continue
        
        # Sort by length and return top chains
        chains.sort(key=lambda x: x['length'], reverse=True)
        return chains[:limit]
    
    def find_models_without_documentation(self) -> List[Dict[str, Any]]:
        """Find models that lack documentation."""
        undocumented = []
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if node_data.get('node_type') == 'model':
                description = node_data.get('description', '')
                meta = node_data.get('meta', {})
                
                if not description and not meta:
                    undocumented.append({
                        'id': node_id,
                        'name': node_data.get('name'),
                        'schema': node_data.get('schema_name'),
                        'path': node_data.get('path')
                    })
        return undocumented
    
    def find_test_patterns(self) -> Dict[str, Any]:
        """Analyze testing patterns across the project."""
        patterns = {
            'test_types': {},
            'most_tested_models': [],
            'least_tested_schemas': {},
            'test_failure_patterns': {}
        }
        
        # Analyze test types
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if node_data.get('node_type') == 'test':
                test_type = node_data.get('test_type', 'unknown')
                patterns['test_types'][test_type] = patterns['test_types'].get(test_type, 0) + 1
        
        # Find most tested models
        model_test_counts = {}
        for node_id in self.graph.nodes():
            if self.graph.nodes[node_id].get('node_type') == 'model':
                test_count = len([
                    n for n in self.graph.successors(node_id)
                    if self.graph.nodes[n].get('node_type') == 'test'
                ])
                model_test_counts[node_id] = {
                    'name': self.graph.nodes[node_id].get('name'),
                    'test_count': test_count
                }
        
        # Sort and get top 10
        sorted_models = sorted(model_test_counts.items(), key=lambda x: x[1]['test_count'], reverse=True)
        patterns['most_tested_models'] = [
            {'model_id': k, **v} for k, v in sorted_models[:10]
        ]
        
        return patterns
    
    def find_data_lineage_to_source(self, model_id: str) -> Dict[str, Any]:
        """Trace a model back to its original source tables."""
        if model_id not in self.graph:
            return {'error': f'Model {model_id} not found'}
        
        # Find all ancestors
        ancestors = nx.ancestors(self.graph, model_id)
        
        # Filter to sources only
        source_nodes = [
            node for node in ancestors 
            if self.graph.nodes[node].get('node_type') == 'source'
        ]
        
        lineage_data = {
            'model': {
                'id': model_id,
                'name': self.graph.nodes[model_id].get('name')
            },
            'sources': [],
            'paths_to_sources': []
        }
        
        # Get source details
        for source in source_nodes:
            source_data = self.graph.nodes[source]
            lineage_data['sources'].append({
                'id': source,
                'name': source_data.get('name'),
                'source_name': source_data.get('source_name'),
                'schema': source_data.get('schema_name'),
                'table': source_data.get('table_name')
            })
            
            # Find path from source to model
            try:
                path = nx.shortest_path(self.graph, source, model_id)
                model_path = [n for n in path if self.graph.nodes[n].get('node_type') in ['source', 'model']]
                lineage_data['paths_to_sources'].append({
                    'source': source,
                    'path': model_path,
                    'path_names': [self.graph.nodes[n].get('name', 'Unknown') for n in model_path]
                })
            except nx.NetworkXNoPath:
                continue
        
        return lineage_data
    
    def find_model_performance_outliers(self, threshold_multiplier: float = 2.0) -> Dict[str, Any]:
        """Find models with unusual performance characteristics."""
        performance_data = self.analyzer.get_performance_analysis()
        
        if not performance_data['model_performance']:
            return {'error': 'No performance data available'}
        
        # Calculate statistics
        execution_times = [m['avg_execution_time'] for m in performance_data['model_performance']]
        avg_time = sum(execution_times) / len(execution_times)
        
        # Find outliers
        outliers = {
            'slow_models': [],
            'fast_models': [],
            'variable_models': []  # Models with high variance in execution time
        }
        
        for model in performance_data['model_performance']:
            avg_exec_time = model['avg_execution_time']
            max_exec_time = model['max_execution_time']
            min_exec_time = model['min_execution_time']
            
            # Slow models
            if avg_exec_time > avg_time * threshold_multiplier:
                outliers['slow_models'].append({
                    'model_id': model['model_id'],
                    'model_name': model['model_name'],
                    'avg_time': avg_exec_time,
                    'slowness_factor': avg_exec_time / avg_time
                })
            
            # Variable models (high variance)
            if max_exec_time > 0 and min_exec_time > 0:
                variance_ratio = max_exec_time / min_exec_time
                if variance_ratio > 3.0:  # More than 3x difference
                    outliers['variable_models'].append({
                        'model_id': model['model_id'],
                        'model_name': model['model_name'],
                        'variance_ratio': variance_ratio,
                        'min_time': min_exec_time,
                        'max_time': max_exec_time
                    })
        
        # Sort outliers
        outliers['slow_models'].sort(key=lambda x: x['slowness_factor'], reverse=True)
        outliers['variable_models'].sort(key=lambda x: x['variance_ratio'], reverse=True)
        
        return outliers
    
    def find_schema_dependencies(self) -> Dict[str, Any]:
        """Analyze dependencies between different schemas."""
        schema_deps = {}
        
        # Build schema dependency graph
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if node_data.get('node_type') == 'model':
                source_schema = node_data.get('schema_name')
                if not source_schema:
                    continue
                
                # Find dependencies
                for pred in self.graph.predecessors(node_id):
                    pred_data = self.graph.nodes[pred]
                    if pred_data.get('node_type') == 'model':
                        target_schema = pred_data.get('schema_name')
                        if target_schema and target_schema != source_schema:
                            if source_schema not in schema_deps:
                                schema_deps[source_schema] = set()
                            schema_deps[source_schema].add(target_schema)
        
        # Convert sets to lists for JSON serialization
        schema_deps = {k: list(v) for k, v in schema_deps.items()}
        
        return {
            'schema_dependencies': schema_deps,
            'cross_schema_count': len(schema_deps),
            'most_dependent_schemas': sorted(
                schema_deps.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )[:5]
        }


def main():
    """Run advanced query examples."""
    print("🔍 dbt Elementary Knowledge Graph - Advanced Queries")
    print("=" * 60)
    
    # Load the graph
    print("\n📂 Loading knowledge graph...")
    storage_manager = GraphStorageManager(backend_type='filesystem')
    
    try:
        graph = storage_manager.load_graph()
        if not graph:
            print("❌ No graph found. Run basic_usage.py first to create a graph.")
            return
        
        print(f"✅ Graph loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    except Exception as e:
        print(f"❌ Failed to load graph: {e}")
        return
    
    # Initialize advanced queries
    advanced_queries = AdvancedGraphQueries(graph)
    
    # Example 1: Find models by tag
    print("\n1️⃣ Finding models by tag...")
    try:
        # You can modify this to search for actual tags in your data
        models_with_mart_tag = advanced_queries.find_models_by_tag('mart')
        print(f"   📊 Models with 'mart' tag: {len(models_with_mart_tag)}")
        for model in models_with_mart_tag[:3]:
            print(f"      - {model['name']} ({model['schema']})")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # Example 2: Find longest dependency chains
    print("\n2️⃣ Finding longest dependency chains...")
    try:
        long_chains = advanced_queries.find_longest_dependency_chains(3)
        print(f"   🔗 Found {len(long_chains)} long chains")
        for i, chain in enumerate(long_chains):
            print(f"      {i+1}. Chain length {chain['length']}: {' → '.join(chain['model_names'][:3])}...")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # Example 3: Schema dependencies
    print("\n3️⃣ Analyzing schema dependencies...")
    try:
        schema_deps = advanced_queries.find_schema_dependencies()
        print(f"   🏗️ Cross-schema dependencies: {schema_deps['cross_schema_count']}")
        for schema, deps in schema_deps['most_dependent_schemas']:
            print(f"      - {schema} depends on {len(deps)} other schemas")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # Example 4: Test patterns
    print("\n4️⃣ Analyzing test patterns...")
    try:
        test_patterns = advanced_queries.find_test_patterns()
        print(f"   🧪 Test type distribution:")
        for test_type, count in test_patterns['test_types'].items():
            print(f"      - {test_type}: {count}")
        
        print(f"   🏆 Most tested models:")
        for model in test_patterns['most_tested_models'][:3]:
            print(f"      - {model['name']}: {model['test_count']} tests")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # Example 5: Performance outliers
    print("\n5️⃣ Finding performance outliers...")
    try:
        outliers = advanced_queries.find_model_performance_outliers()
        if 'error' not in outliers:
            print(f"   🐌 Slow models: {len(outliers['slow_models'])}")
            for model in outliers['slow_models'][:3]:
                print(f"      - {model['model_name']}: {model['slowness_factor']:.1f}x slower than average")
            
            print(f"   📊 Variable models: {len(outliers['variable_models'])}")
            for model in outliers['variable_models'][:3]:
                print(f"      - {model['model_name']}: {model['variance_ratio']:.1f}x variance")
        else:
            print(f"   ⚠️ {outliers['error']}")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # Example 6: Data lineage to source
    print("\n6️⃣ Tracing data lineage to sources...")
    try:
        # Get a sample model
        model_nodes = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'model']
        if model_nodes:
            sample_model = model_nodes[0]
            lineage = advanced_queries.find_data_lineage_to_source(sample_model)
            
            if 'error' not in lineage:
                print(f"   📊 Model: {lineage['model']['name']}")
                print(f"   🔍 Traces back to {len(lineage['sources'])} sources:")
                for source in lineage['sources'][:3]:
                    print(f"      - {source['source_name']}.{source['name']}")
            else:
                print(f"   ⚠️ {lineage['error']}")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    print("\n🎉 Advanced queries completed!")
    print("\n💡 Try modifying these queries for your specific use cases:")
    print("   - Search for models with specific patterns")
    print("   - Analyze test coverage by business domain")
    print("   - Find models that haven't been updated recently")
    print("   - Identify potential refactoring opportunities")


if __name__ == "__main__":
    main()