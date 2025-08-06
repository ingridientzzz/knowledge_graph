"""
Basic usage example for the dbt Elementary Knowledge Graph.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_extractor import ElementaryDataExtractor
from src.graph_builder import KnowledgeGraphBuilder
from src.graph_storage import GraphStorageManager
from src.graph_analyzer import GraphAnalyzer


def main():
    """Main example function."""
    print("🚀 dbt Elementary Knowledge Graph - Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Extract data from Snowflake Elementary tables
    print("\n1️⃣ Extracting data from Snowflake...")
    
    try:
        # Initialize extractor - using your elementary_log schema
        extractor = ElementaryDataExtractor(profile_name='test_project', target='test_project')
        extractor.set_elementary_schema('elementary_log')
        
        # Extract all Elementary data
        data = extractor.extract_all_data()
        print(f"✅ Successfully extracted data from {len(data)} table types")
        
        # Show sample data
        for table_name, df in data.items():
            if not df.empty:
                print(f"   📊 {table_name}: {len(df)} rows")
                print(f"      Sample columns: {list(df.columns)[:5]}")
    
    except Exception as e:
        print(f"❌ Failed to extract data: {e}")
        print("💡 Make sure your ~/.dbt/profiles.yml is configured with Snowflake connection")
        print("💡 And that you have Elementary tables in your Snowflake database")
        return
    
    # Step 2: Build the knowledge graph
    print("\n2️⃣ Building knowledge graph...")
    builder = KnowledgeGraphBuilder()
    
    try:
        graph = builder.build_graph_from_data(data)
        print(f"✅ Knowledge graph built successfully!")
        print(f"   📈 Graph contains {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    except Exception as e:
        print(f"❌ Failed to build graph: {e}")
        return
    
    # Step 3: Save the graph
    print("\n3️⃣ Saving knowledge graph...")
    storage_manager = GraphStorageManager(backend_type='filesystem')
    
    try:
        success = storage_manager.save_graph(graph)
        if success:
            print("✅ Graph saved successfully to filesystem")
        else:
            print("❌ Failed to save graph")
    
    except Exception as e:
        print(f"❌ Error saving graph: {e}")
    
    # Step 4: Analyze the graph
    print("\n4️⃣ Analyzing knowledge graph...")
    analyzer = GraphAnalyzer(graph)
    
    try:
        # Get models for analysis
        model_nodes = [node for node in graph.nodes() if graph.nodes[node].get('node_type') == 'model']
        
        if model_nodes:
            # Sample model for detailed analysis
            sample_model = model_nodes[0]
            print(f"\n🔍 Analyzing sample model: {graph.nodes[sample_model].get('name', 'Unknown')}")
            
            # Get lineage
            lineage = analyzer.get_model_lineage(sample_model)
            print(f"   ⬆️  Upstream dependencies: {len(lineage['upstream'])}")
            print(f"   ⬇️  Downstream dependencies: {len(lineage['downstream'])}")
            
            # Impact analysis
            impact = analyzer.get_impact_analysis(sample_model)
            print(f"   💥 Total nodes impacted by changes: {impact.get('total_affected', 0)}")
        
        # Test coverage analysis
        coverage = analyzer.get_test_coverage()
        print(f"\n🧪 Test Coverage Analysis:")
        print(f"   📊 Total models: {coverage['total_models']}")
        print(f"   ✅ Models with tests: {coverage['covered_models']}")
        print(f"   📈 Coverage percentage: {coverage['coverage_percentage']:.1f}%")
        
        # Data quality issues
        issues = analyzer.find_data_quality_issues()
        print(f"\n🚨 Data Quality Issues:")
        print(f"   ❌ Failing tests: {issues['summary']['total_failing_tests']}")
        print(f"   ⚠️  Models without tests: {issues['summary']['models_without_tests_count']}")
        
        # Performance analysis
        performance = analyzer.get_performance_analysis()
        if performance['model_performance']:
            print(f"\n⚡ Performance Analysis:")
            slowest_model = performance['model_performance'][0]
            print(f"   🐌 Slowest model: {slowest_model['model_name']} ({slowest_model['avg_execution_time']:.2f}s avg)")
        
        # Dependency complexity
        complexity = analyzer.get_dependency_complexity()
        if complexity['summary']:
            print(f"\n🕸️  Dependency Complexity:")
            print(f"   📊 Average dependencies per model: {complexity['summary']['avg_dependencies_per_model']:.1f}")
            print(f"   🔗 Models with high complexity: {complexity['summary']['models_with_high_complexity']}")
            print(f"   🔄 Circular dependencies: {complexity['summary']['circular_dependencies_count']}")
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
    
    # Step 5: Example queries
    print("\n5️⃣ Example Graph Queries...")
    
    try:
        # Query specific node types
        model_count = len([n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'model'])
        test_count = len([n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'test'])
        source_count = len([n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'source'])
        
        print(f"   📦 Models: {model_count}")
        print(f"   🧪 Tests: {test_count}")
        print(f"   📊 Sources: {source_count}")
        
        # Find most connected nodes
        degrees = dict(graph.degree())
        if degrees:
            most_connected = max(degrees, key=degrees.get)
            print(f"   🌟 Most connected node: {graph.nodes[most_connected].get('name', 'Unknown')} ({degrees[most_connected]} connections)")
    
    except Exception as e:
        print(f"❌ Error during queries: {e}")
    
    print("\n🎉 Basic usage example completed!")
    print("\n💡 Next steps:")
    print("   - Explore the graph_analyzer.py for more advanced queries")
    print("   - Check out visualization examples in the notebooks/ folder")
    print("   - Set up Neo4j for more powerful graph queries")


if __name__ == "__main__":
    main()