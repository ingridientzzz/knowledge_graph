#!/usr/bin/env python3
"""
Quick test of graph building with extracted data.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_extractor import ElementaryDataExtractor
from src.graph_builder import KnowledgeGraphBuilder


def main():
    """Test graph building."""
    print("🧪 Quick Graph Build Test")
    print("=" * 30)
    
    try:
        # Extract data
        print("\n1️⃣ Extracting data...")
        extractor = ElementaryDataExtractor(profile_name='test_project', target='test_project')
        extractor.set_elementary_schema('elementary_log')
        
        # Just extract the working tables for now
        extractor.connect()
        
        # Get a small subset for testing
        models_df = extractor.extract_models()
        tests_df = extractor.extract_tests() 
        sources_df = extractor.extract_sources()
        test_results_df = extractor.extract_test_results()
        
        extractor.disconnect()
        
        print(f"✅ Extracted: {len(models_df)} models, {len(tests_df)} tests, {len(sources_df)} sources, {len(test_results_df)} test results")
        
        if len(models_df) > 0:
            print(f"📋 Sample model columns: {list(models_df.columns)}")
            print(f"📋 First model unique_id: {models_df.iloc[0]['unique_id'] if 'unique_id' in models_df.columns else 'MISSING'}")
        
        # Test graph building
        print("\n2️⃣ Building graph...")
        builder = KnowledgeGraphBuilder()
        
        data = {
            'models': models_df,
            'tests': tests_df,
            'sources': sources_df,
            'test_results': test_results_df,
            'run_results': extractor.extract_run_results(),  # This might fail but that's OK
            'invocations': extractor.extract_invocations(),  # This might fail but that's OK
            'model_columns': extractor.extract_model_columns()  # Empty
        }
        
        graph = builder.build_graph_from_data(data)
        
        print(f"\n✅ Graph built successfully!")
        print(f"   📊 Nodes: {len(graph.nodes)}")
        print(f"   🔗 Edges: {len(graph.edges)}")
        
        # Show some sample nodes
        print(f"\n📋 Sample nodes:")
        for i, (node_id, node_attrs) in enumerate(list(graph.nodes(data=True))[:5]):
            node_type = node_attrs.get('node_type', 'unknown')
            name = node_attrs.get('name', 'unnamed')
            print(f"   {i+1}. {node_type}: {name} ({node_id[:50]}...)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")