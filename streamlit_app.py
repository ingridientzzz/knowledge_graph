#!/usr/bin/env python3
"""
Streamlit app for visualizing the dbt Elementary Knowledge Graph.
"""
import streamlit as st
import pandas as pd
import networkx as nx
import json
import pickle
from pathlib import Path
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge, Config

# Set page config
st.set_page_config(
    page_title="dbt Elementary Knowledge Graph",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define color scheme
COLORS = {
    "model": "#4287f5",  # blue
    "test": "#42f59e",   # green
    "source": "#f5a142",  # orange
    "column": "#f542e5",  # pink
    "invocation": "#f54242",  # red
    "snapshot": "#b042f5",  # purple
    "unknown": "#999999",  # gray
}

@st.cache_data
def load_graph_from_pickle():
    """Load the graph from pickle file."""
    try:
        pickle_path = Path("data/knowledge_graph.gpickle")
        with open(pickle_path, 'rb') as f:
            graph = pickle.load(f)
        return graph
    except Exception as e:
        st.error(f"Failed to load graph from pickle: {e}")
        return None

@st.cache_data
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
        
        return graph
    except Exception as e:
        st.error(f"Failed to load graph from JSON: {e}")
        return None

@st.cache_data
def load_metadata():
    """Load graph metadata."""
    try:
        metadata_path = Path("data/graph_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        st.warning(f"Failed to load metadata: {e}")
        return {}

def get_node_data(graph):
    """Extract node data from graph for display."""
    nodes_data = []
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        name = attrs.get('name', node_id)
        
        # Get properties
        properties = attrs.get('properties', {})
        
        # Create a row for the dataframe
        row = {
            'id': node_id,
            'name': name,
            'type': node_type,
        }
        
        # Add properties
        for k, v in properties.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v
        
        nodes_data.append(row)
    
    return pd.DataFrame(nodes_data)

def get_edge_data(graph):
    """Extract edge data from graph for display."""
    edges_data = []
    for source, target, attrs in graph.edges(data=True):
        edge_type = attrs.get('edge_type', 'related')
        
        # Get properties
        properties = attrs.get('properties', {})
        
        # Create a row for the dataframe
        row = {
            'source': source,
            'target': target,
            'type': edge_type,
        }
        
        # Add properties
        for k, v in properties.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v
        
        edges_data.append(row)
    
    return pd.DataFrame(edges_data)

def create_agraph_visualization(graph, selected_node_types=None, selected_edge_types=None, 
                               search_query=None, limit=100):
    """Create visualization using streamlit-agraph."""
    
    # Filter nodes based on selected types and search query
    filtered_nodes = []
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        name = attrs.get('name', node_id)
        
        # Apply filters
        if selected_node_types and node_type not in selected_node_types:
            continue
            
        if search_query and search_query.lower() not in name.lower() and search_query.lower() not in node_id.lower():
            continue
            
        filtered_nodes.append(node_id)
    
    # Limit the number of nodes to avoid performance issues
    if len(filtered_nodes) > limit:
        st.warning(f"Too many nodes to display ({len(filtered_nodes)}). Showing first {limit} nodes.")
        filtered_nodes = filtered_nodes[:limit]
    
    # Create subgraph with filtered nodes
    subgraph = graph.subgraph(filtered_nodes)
    
    # Create nodes for agraph
    nodes = []
    for node_id, attrs in subgraph.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        name = attrs.get('name', node_id)
        
        # Create tooltip with properties
        tooltip = f"<b>{name}</b> ({node_type})<br>"
        if 'properties' in attrs:
            for k, v in attrs['properties'].items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    tooltip += f"{k}: {v}<br>"
        
        # Create node
        nodes.append(Node(
            id=node_id,
            label=name,
            size=20 if node_type == 'model' else 15,
            color=COLORS.get(node_type, COLORS['unknown']),
            shape="dot" if node_type == 'model' else "triangle" if node_type == 'test' else "square",
            title=tooltip
        ))
    
    # Create edges for agraph
    edges = []
    for source, target, attrs in subgraph.edges(data=True):
        edge_type = attrs.get('edge_type', 'related')
        
        # Apply edge type filter
        if selected_edge_types and edge_type not in selected_edge_types:
            continue
            
        # Create tooltip with properties
        tooltip = f"<b>{edge_type}</b><br>"
        if 'properties' in attrs:
            for k, v in attrs['properties'].items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    tooltip += f"{k}: {v}<br>"
        
        # Create edge
        edges.append(Edge(
            source=source,
            target=target,
            label=edge_type,
            title=tooltip,
            color="#666666",
            type="CURVE_SMOOTH"
        ))
    
    # Configuration
    config = Config(
        width=1200,
        height=800,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
    )
    
    return nodes, edges, config

def create_plotly_visualization(graph, selected_nodes=None, limit=50):
    """Create a plotly visualization of the graph."""
    if selected_nodes is None or len(selected_nodes) == 0:
        # Get top nodes by degree
        top_nodes = sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True)[:limit]
        subgraph = graph.subgraph(top_nodes)
    else:
        # Get subgraph with selected nodes and their neighbors
        neighbors = set()
        for node in selected_nodes:
            if node in graph:
                neighbors.update(graph.neighbors(node))
        
        nodes_to_include = set(selected_nodes) | neighbors
        if len(nodes_to_include) > limit:
            st.warning(f"Too many nodes to display ({len(nodes_to_include)}). Showing first {limit} nodes.")
            nodes_to_include = list(nodes_to_include)[:limit]
            
        subgraph = graph.subgraph(nodes_to_include)
    
    # Create layout
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
    
    # Create node traces
    node_traces = {}
    for node_type in set(nx.get_node_attributes(subgraph, 'node_type').values()):
        node_traces[node_type] = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=COLORS.get(node_type, COLORS['unknown']),
                size=15,
                line=dict(width=1, color='#888')
            ),
            name=node_type
        )
    
    # Add nodes to traces
    for node, attrs in subgraph.nodes(data=True):
        x, y = pos[node]
        node_type = attrs.get('node_type', 'unknown')
        name = attrs.get('name', node)
        
        # Create hover text
        hover_text = f"<b>{name}</b> ({node_type})<br>"
        if 'properties' in attrs:
            for k, v in attrs['properties'].items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    hover_text += f"{k}: {v}<br>"
        
        node_traces[node_type].x = node_traces[node_type].x + (x,)
        node_traces[node_type].y = node_traces[node_type].y + (y,)
        node_traces[node_type].text = node_traces[node_type].text + (hover_text,)
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Add edges
    for u, v, attrs in subgraph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace.x += (x0, x1, None)
        edge_trace.y += (y0, y1, None)
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace] + list(node_traces.values()),
        layout=go.Layout(
            title='dbt Elementary Knowledge Graph',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800
        )
    )
    
    return fig

def main():
    """Main Streamlit app."""
    st.title("dbt Elementary Knowledge Graph Explorer")
    
    # Sidebar
    st.sidebar.title("Options")
    
    # Load data
    graph = load_graph_from_pickle()
    if graph is None:
        graph = load_graph_from_json()
        
    if graph is None:
        st.error("Failed to load graph data. Please check the data files.")
        return
    
    metadata = load_metadata()
    
    # Display metadata
    st.sidebar.subheader("Graph Info")
    st.sidebar.write(f"Nodes: {len(graph.nodes)}")
    st.sidebar.write(f"Edges: {len(graph.edges)}")
    
    if metadata:
        st.sidebar.write(f"Created: {metadata.get('created_at', 'Unknown')}")
    
    # Count node types
    node_types = {}
    for _, attrs in graph.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    # Count edge types
    edge_types = {}
    for _, _, attrs in graph.edges(data=True):
        edge_type = attrs.get('edge_type', 'related')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    # Display counts
    st.sidebar.subheader("Node Types")
    for node_type, count in node_types.items():
        st.sidebar.write(f"{node_type}: {count}")
    
    st.sidebar.subheader("Edge Types")
    for edge_type, count in edge_types.items():
        st.sidebar.write(f"{edge_type}: {count}")
    
    # Filters
    st.sidebar.subheader("Filters")
    
    # Node type filter
    selected_node_types = st.sidebar.multiselect(
        "Node Types",
        options=list(node_types.keys()),
        default=list(node_types.keys())
    )
    
    # Edge type filter
    selected_edge_types = st.sidebar.multiselect(
        "Edge Types",
        options=list(edge_types.keys()),
        default=list(edge_types.keys())
    )
    
    # Search filter
    search_query = st.sidebar.text_input("Search Nodes", "")
    
    # Node limit
    node_limit = st.sidebar.slider("Max Nodes", 10, 200, 100)
    
    # Visualization type
    viz_type = st.sidebar.radio("Visualization Type", ["Interactive Graph", "Network Graph", "Data Tables"])
    
    # Main content
    if viz_type == "Interactive Graph":
        st.header("Interactive Knowledge Graph")
        
        nodes, edges, config = create_agraph_visualization(
            graph,
            selected_node_types=selected_node_types,
            selected_edge_types=selected_edge_types,
            search_query=search_query,
            limit=node_limit
        )
        
        if len(nodes) > 0:
            st.write(f"Displaying {len(nodes)} nodes and {len(edges)} edges")
            agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.warning("No nodes match the current filters.")
    
    elif viz_type == "Network Graph":
        st.header("Network Graph")
        
        # Get selected nodes based on filters
        selected_nodes = []
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            name = attrs.get('name', node_id)
            
            if selected_node_types and node_type not in selected_node_types:
                continue
                
            if search_query and search_query.lower() not in name.lower() and search_query.lower() not in node_id.lower():
                continue
                
            selected_nodes.append(node_id)
        
        # Create plotly visualization
        fig = create_plotly_visualization(graph, selected_nodes=selected_nodes, limit=node_limit)
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Data Tables
        st.header("Knowledge Graph Data")
        
        # Extract node and edge data
        nodes_df = get_node_data(graph)
        edges_df = get_edge_data(graph)
        
        # Apply filters
        if selected_node_types:
            nodes_df = nodes_df[nodes_df['type'].isin(selected_node_types)]
            
        if search_query:
            nodes_df = nodes_df[
                nodes_df['name'].str.contains(search_query, case=False, na=False) |
                nodes_df['id'].str.contains(search_query, case=False, na=False)
            ]
            
        if selected_edge_types:
            edges_df = edges_df[edges_df['type'].isin(selected_edge_types)]
        
        # Display tables
        tab1, tab2 = st.tabs(["Nodes", "Edges"])
        
        with tab1:
            st.write(f"Showing {len(nodes_df)} nodes")
            st.dataframe(nodes_df, use_container_width=True)
            
        with tab2:
            st.write(f"Showing {len(edges_df)} edges")
            st.dataframe(edges_df, use_container_width=True)


if __name__ == "__main__":
    main()