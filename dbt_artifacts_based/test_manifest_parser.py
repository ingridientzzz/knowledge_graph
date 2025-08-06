#!/usr/bin/env python3
"""
Test script for the manifest parser.
Demonstrates parsing a manifest.json file and creating a knowledge graph.
"""

import logging
from pathlib import Path
from manifest_parser import ManifestParser
from graph_storage import GraphStorageManager


def test_manifest_parsing():
    """Test the manifest parsing functionality."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Paths
    manifest_path = "code_docs/manifest.json"
    output_dir = "data"
    
    logger.info("Starting manifest parser test...")
    
    if not Path(manifest_path).exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        return
    
    try:
        # Test 1: Direct manifest parsing
        logger.info("=" * 50)
        logger.info("TEST 1: Direct Manifest Parsing")
        logger.info("=" * 50)
        
        parser = ManifestParser(manifest_path)
        nodes, edges = parser.parse_manifest()
        
        # Print parsing results
        stats = parser.get_statistics()
        print(f"\n📊 Parsing Results:")
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Total Edges: {stats['total_edges']}")
        print(f"dbt Version: {stats['manifest_version']}")
        
        print(f"\n📋 Node Breakdown:")
        for node_type, count in stats['node_counts'].items():
            if count > 0:
                print(f"  {node_type.ljust(12)}: {count}")
        
        print(f"\n🔗 Edge Breakdown:")
        for edge_type, count in stats['edge_counts'].items():
            if count > 0:
                print(f"  {edge_type.ljust(12)}: {count}")
        
        # Test 2: Graph storage integration
        logger.info("\n" + "=" * 50)
        logger.info("TEST 2: Graph Storage Integration")
        logger.info("=" * 50)
        
        storage_manager = GraphStorageManager(output_dir)
        graph = storage_manager.build_graph_from_manifest(manifest_path)
        
        print(f"\n🗄️  Storage Results:")
        print(f"NetworkX Graph Nodes: {graph.number_of_nodes()}")
        print(f"NetworkX Graph Edges: {graph.number_of_edges()}")
        
        # Test graph loading
        loaded_graph = storage_manager.load_graph()
        if loaded_graph:
            print(f"✅ Graph successfully loaded from storage")
            print(f"Loaded Nodes: {loaded_graph.number_of_nodes()}")
            print(f"Loaded Edges: {loaded_graph.number_of_edges()}")
        else:
            print("❌ Failed to load graph from storage")
        
        # Test 3: Sample node analysis
        logger.info("\n" + "=" * 50)
        logger.info("TEST 3: Sample Node Analysis")
        logger.info("=" * 50)
        
        # Show sample nodes of each type
        node_samples = {}
        for node in list(nodes.values())[:50]:  # Limit to first 50 for performance
            node_type = node.node_type.value
            if node_type not in node_samples:
                node_samples[node_type] = []
            if len(node_samples[node_type]) < 3:  # Show up to 3 examples per type
                node_samples[node_type].append(node)
        
        for node_type, sample_nodes in node_samples.items():
            print(f"\n📌 Sample {node_type.upper()} nodes:")
            for node in sample_nodes:
                print(f"  • {node.name} ({node.unique_id})")
                # Show a few key properties
                key_props = ['description', 'package_name', 'resource_type']
                props = []
                for prop in key_props:
                    if prop in node.properties and node.properties[prop]:
                        value = str(node.properties[prop])
                        if len(value) > 50:
                            value = value[:47] + "..."
                        props.append(f"{prop}: {value}")
                if props:
                    print(f"    {' | '.join(props)}")
        
        # Test 4: Sample relationship analysis
        logger.info("\n" + "=" * 50)
        logger.info("TEST 4: Sample Relationship Analysis")
        logger.info("=" * 50)
        
        # Show sample edges of each type
        edge_samples = {}
        for edge in edges[:50]:  # Limit to first 50 for performance
            edge_type = edge.edge_type.value
            if edge_type not in edge_samples:
                edge_samples[edge_type] = []
            if len(edge_samples[edge_type]) < 3:  # Show up to 3 examples per type
                edge_samples[edge_type].append(edge)
        
        for edge_type, sample_edges in edge_samples.items():
            print(f"\n🔗 Sample {edge_type.upper()} relationships:")
            for edge in sample_edges:
                source_name = nodes.get(edge.source_id, type('obj', (object,), {'name': edge.source_id})).name
                target_name = nodes.get(edge.target_id, type('obj', (object,), {'name': edge.target_id})).name
                print(f"  • {source_name} → {target_name}")
        
        # Test 5: Export visualization data
        logger.info("\n" + "=" * 50)
        logger.info("TEST 5: Visualization Export")
        logger.info("=" * 50)
        
        exported_files = storage_manager.export_for_visualization("visualization")
        print(f"\n📤 Exported visualization files:")
        for format_name, file_path in exported_files.items():
            print(f"  {format_name}: {file_path}")
        
        logger.info("\n✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    test_manifest_parsing()