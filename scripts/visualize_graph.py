#!/usr/bin/env python3
"""
Quick visualization script for the knowledge graph.
Creates an interactive HTML visualization using PyVis.
"""
import sys
import os
import pickle
import json
from pathlib import Path
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def load_graph_from_pickle():
    """Load the graph from pickle file."""
    try:
        pickle_path = Path("data/knowledge_graph.gpickle")
        with open(pickle_path, 'rb') as f:
            graph = pickle.load(f)
        print(f"✅ Loaded graph from {pickle_path} with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
    except Exception as e:
        print(f"❌ Failed to load graph from pickle: {e}")
        return None


def load_graph_from_json():
    """Load the graph from JSON files."""
    try:
        nodes_path = Path("data/nodes.json")
        edges_path = Path("data/edges.json")
        
        with open(nodes_path, 'r') as f:
            nodes_data = json.load(f)
        
        with open(edges_path, 'r') as f:
            edges_data = json.load(f)
        
        graph = nx.MultiDiGraph()
        
        # Add nodes
        for node in nodes_data:
            node_id = node.get('id')
            if node_id:
                # Remove id from attributes since it's used as the node identifier
                attrs = {k: v for k, v in node.items() if k != 'id'}
                graph.add_node(node_id, **attrs)
        
        # Add edges
        for edge in edges_data:
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                # Remove source/target from attributes
                attrs = {k: v for k, v in edge.items() if k not in ('source', 'target')}
                graph.add_edge(source, target, **attrs)
        
        print(f"✅ Loaded graph from JSON with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
    except Exception as e:
        print(f"❌ Failed to load graph from JSON: {e}")
        return None


def create_interactive_visualization(graph, output_path="graph_visualization.html", height="800px", width="100%"):
    """Create an interactive visualization using PyVis."""
    try:
        # Create a PyVis network
        net = Network(height=height, width=width, directed=True, notebook=False)
        
        # Define node colors by type
        node_colors = {
            'model': '#4287f5',  # blue
            'test': '#42f59e',   # green
            'source': '#f5a142',  # orange
            'column': '#f542e5',  # pink
            'invocation': '#f54242',  # red
            'snapshot': '#b042f5'  # purple
        }
        
        # Add nodes with proper styling
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            name = attrs.get('name', str(node_id)[:20])
            
            # Create a nice title with properties
            title = f"<b>{name}</b> ({node_type})<br>"
            if 'properties' in attrs:
                for k, v in attrs['properties'].items():
                    if isinstance(v, (dict, list)):
                        # Skip complex properties in the tooltip
                        continue
                    title += f"{k}: {v}<br>"
            
            # Add the node with styling
            net.add_node(
                node_id, 
                label=name, 
                title=title, 
                color=node_colors.get(node_type, '#999999'),
                shape='dot' if node_type == 'model' else 'triangle' if node_type == 'test' else 'square',
                size=20 if node_type == 'model' else 15
            )
        
        # Add edges with proper styling
        for source, target, attrs in graph.edges(data=True):
            edge_type = attrs.get('edge_type', 'related')
            
            # Create edge label and title
            title = f"<b>{edge_type}</b><br>"
            if 'properties' in attrs:
                for k, v in attrs['properties'].items():
                    if isinstance(v, (dict, list)):
                        # Skip complex properties in the tooltip
                        continue
                    title += f"{k}: {v}<br>"
            
            # Add the edge with styling
            net.add_edge(
                source, 
                target, 
                title=title,
                label=edge_type,
                arrows='to',
                color='#666666'
            )
        
        # Configure physics for better layout
        net.barnes_hut(
            gravity=-80000,
            central_gravity=0.3,
            spring_length=250,
            spring_strength=0.001,
            damping=0.09,
            overlap=0
        )
        
        # Set other options
        net.set_options("""
        var options = {
          "nodes": {
            "font": {
              "size": 12,
              "face": "Tahoma"
            }
          },
          "edges": {
            "font": {
              "size": 10,
              "align": "middle"
            },
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            }
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "iterations": 1000
            }
          }
        }
        """)
        
        # Save the visualization
        net.save_graph(output_path)
        print(f"✅ Interactive visualization saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create visualization: {e}")
        return False


def create_matplotlib_visualization(graph, output_path="graph_static.png"):
    """Create a static visualization using matplotlib."""
    try:
        # Create a smaller subgraph for better visualization
        # Get the top 20 most connected nodes
        top_nodes = sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True)[:20]
        subgraph = graph.subgraph(top_nodes)
        
        plt.figure(figsize=(12, 10))
        
        # Define node colors by type
        node_colors = []
        for node, attrs in subgraph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            if node_type == 'model':
                node_colors.append('skyblue')
            elif node_type == 'test':
                node_colors.append('lightgreen')
            elif node_type == 'source':
                node_colors.append('orange')
            else:
                node_colors.append('gray')
        
        # Create layout
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos, node_size=700, node_color=node_colors, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5, arrows=True)
        
        # Draw labels
        labels = {}
        for node, attrs in subgraph.nodes(data=True):
            labels[node] = attrs.get('name', str(node)[:10])
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        plt.title("dbt Elementary Knowledge Graph (Top 20 Connected Nodes)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"✅ Static visualization saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create static visualization: {e}")
        return False


def main():
    """Main function to create visualizations."""
    print("🎨 Knowledge Graph Visualization")
    print("=" * 35)
    
    # Try loading from pickle first, then JSON if that fails
    graph = load_graph_from_pickle()
    if graph is None:
        graph = load_graph_from_json()
    
    if graph is None:
        print("❌ Failed to load graph from any source")
        return False
    
    # Print graph info
    print("\n📊 Graph Statistics:")
    print(f"   - Nodes: {len(graph.nodes)}")
    print(f"   - Edges: {len(graph.edges)}")
    
    # Count node types
    node_types = {}
    for _, attrs in graph.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\n📋 Node Types:")
    for node_type, count in node_types.items():
        print(f"   - {node_type}: {count}")
    
    # Create visualizations
    print("\n🖌️ Creating visualizations...")
    interactive_success = create_interactive_visualization(graph)
    static_success = create_matplotlib_visualization(graph)
    
    if interactive_success:
        print("\n🎉 Interactive visualization created successfully!")
        print("   Open graph_visualization.html in your browser to explore the graph")
    
    if static_success:
        print("\n🖼️ Static visualization created successfully!")
        print("   Open graph_static.png to see a preview of the top connected nodes")
    
    return interactive_success or static_success


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💥 Visualization creation failed")
    else:
        print("\n✅ Done!")