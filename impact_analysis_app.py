#!/usr/bin/env python3
"""
Streamlit app for dbt model impact analysis.
Allows users to select a model and visualize upstream and downstream dependencies.
"""
import streamlit as st
import pandas as pd
import networkx as nx
import json
import pickle
from pathlib import Path
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge, Config

# Add project root to path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph_storage import GraphStorageManager
from src.graph_analyzer import GraphAnalyzer
from src.graph_entities import NodeType, EdgeType

# Set page config
st.set_page_config(
    page_title="dbt Model Impact Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define color scheme
COLORS = {
    "model": "#4287f5",          # blue
    "test": "#42f59e",           # green
    "source": "#f5a142",         # orange
    "column": "#f542e5",         # pink
    "invocation": "#f54242",     # red
    "snapshot": "#b042f5",       # purple
    "unknown": "#999999",        # gray
    "selected": "#ff0000",       # bright red for selected model
    "upstream": "#ff9900",       # orange for upstream dependencies
    "downstream": "#0099ff",     # bright blue for downstream dependencies
}

@st.cache_data
def load_graph():
    """Load the graph from pickle or JSON files."""
    # Try pickle first
    storage_manager = GraphStorageManager(backend_type='filesystem')
    graph = storage_manager.load_graph()
    if graph:
        return graph
    
    # Fall back to JSON if pickle fails
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
                attrs = {k: v for k, v in node.items() if k != 'id'}
                graph.add_node(node_id, **attrs)
        
        # Add edges
        for edge in edges_data:
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                attrs = {k: v for k, v in edge.items() if k not in ('source', 'target')}
                graph.add_edge(source, target, **attrs)
        
        return graph
    except Exception as e:
        st.error(f"Failed to load graph data: {e}")
        return None

def get_model_list(graph):
    """Get list of models from the graph."""
    models = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('node_type') == NodeType.MODEL.value:
            name = attrs.get('name', node_id)
            models.append((node_id, name))
    
    # Sort by name
    return sorted(models, key=lambda x: x[1])

def get_upstream_dependencies(graph, node_id, depth=None):
    """Get upstream dependencies (parents) of a node."""
    if depth == 0:
        return set()
    
    upstream = set()
    for predecessor in graph.predecessors(node_id):
        upstream.add(predecessor)
        if depth is None or depth > 1:
            upstream.update(get_upstream_dependencies(graph, predecessor, None if depth is None else depth - 1))
    
    return upstream

def get_downstream_dependencies(graph, node_id, depth=None):
    """Get downstream dependencies (children) of a node."""
    if depth == 0:
        return set()
    
    downstream = set()
    for successor in graph.successors(node_id):
        downstream.add(successor)
        if depth is None or depth > 1:
            downstream.update(get_downstream_dependencies(graph, successor, None if depth is None else depth - 1))
    
    return downstream

def create_impact_subgraph(graph, selected_model, upstream_depth, downstream_depth, include_tests):
    """Create a subgraph showing the impact of the selected model."""
    # Get upstream and downstream nodes
    upstream_nodes = get_upstream_dependencies(graph, selected_model, upstream_depth)
    downstream_nodes = get_downstream_dependencies(graph, selected_model, downstream_depth)
    
    # Create node set
    nodes_to_include = {selected_model} | upstream_nodes | downstream_nodes
    
    # Filter out tests if not included
    if not include_tests:
        nodes_to_include = {
            node_id for node_id in nodes_to_include
            if graph.nodes[node_id].get('node_type') != NodeType.TEST.value
        }
    
    # Create subgraph
    subgraph = graph.subgraph(nodes_to_include)
    
    return subgraph, upstream_nodes, downstream_nodes

def create_impact_visualization(graph, subgraph, selected_model, upstream_nodes, downstream_nodes):
    """Create interactive visualization of the impact graph."""
    # Create nodes for agraph
    nodes = []
    for node_id, attrs in subgraph.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        name = attrs.get('name', node_id)
        
        # Determine node color based on impact
        if node_id == selected_model:
            color = COLORS['selected']
            size = 30
        elif node_id in upstream_nodes:
            color = COLORS['upstream']
            size = 20
        elif node_id in downstream_nodes:
            color = COLORS['downstream']
            size = 20
        else:
            color = COLORS.get(node_type, COLORS['unknown'])
            size = 15
        
        # Create tooltip with properties
        tooltip = f"<b>{name}</b> ({node_type})<br>"
        for k, v in attrs.items():
            if k not in ['node_type', 'name'] and isinstance(v, (str, int, float, bool)) or v is None:
                tooltip += f"{k}: {v}<br>"
        
        # Create node
        nodes.append(Node(
            id=node_id,
            label=name,
            size=size,
            color=color,
            shape="dot" if node_type == NodeType.MODEL.value else "triangle" if node_type == NodeType.TEST.value else "square",
            title=tooltip
        ))
    
    # Create edges for agraph
    edges = []
    for source, target, attrs in subgraph.edges(data=True):
        edge_type = attrs.get('edge_type', 'related')
        
        # Determine edge color based on impact
        if source == selected_model:
            color = COLORS['downstream']
            width = 3
        elif target == selected_model:
            color = COLORS['upstream']
            width = 3
        else:
            color = "#666666"
            width = 1
        
        # Create tooltip
        tooltip = f"<b>{edge_type}</b><br>"
        for k, v in attrs.items():
            if k != 'edge_type' and isinstance(v, (str, int, float, bool)) or v is None:
                tooltip += f"{k}: {v}<br>"
        
        # Create edge
        edges.append(Edge(
            source=source,
            target=target,
            label=edge_type,
            title=tooltip,
            color=color,
            width=width,
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

def create_impact_metrics(graph, selected_model, upstream_nodes, downstream_nodes):
    """Create metrics for impact analysis."""
    metrics = {}
    
    # Basic counts
    metrics['upstream_count'] = len(upstream_nodes)
    metrics['downstream_count'] = len(downstream_nodes)
    metrics['total_impact'] = len(upstream_nodes) + len(downstream_nodes) + 1  # +1 for selected model
    
    # Count by node type
    upstream_by_type = {}
    downstream_by_type = {}
    
    for node_id in upstream_nodes:
        node_type = graph.nodes[node_id].get('node_type', 'unknown')
        upstream_by_type[node_type] = upstream_by_type.get(node_type, 0) + 1
    
    for node_id in downstream_nodes:
        node_type = graph.nodes[node_id].get('node_type', 'unknown')
        downstream_by_type[node_type] = downstream_by_type.get(node_type, 0) + 1
    
    metrics['upstream_by_type'] = upstream_by_type
    metrics['downstream_by_type'] = downstream_by_type
    
    # Risk assessment
    risk_level = "Low"
    risk_factors = []
    
    if len(downstream_nodes) > 10:
        risk_level = "High"
        risk_factors.append(f"Large downstream impact ({len(downstream_nodes)} nodes)")
    elif len(downstream_nodes) > 5:
        risk_level = "Medium"
        risk_factors.append(f"Moderate downstream impact ({len(downstream_nodes)} nodes)")
    
    # Check for critical downstream models (those with many dependencies)
    critical_models = []
    for node_id in downstream_nodes:
        if graph.nodes[node_id].get('node_type') == NodeType.MODEL.value:
            downstream_count = len(list(graph.successors(node_id)))
            if downstream_count > 5:
                name = graph.nodes[node_id].get('name', node_id)
                critical_models.append((name, downstream_count))
    
    if critical_models:
        if risk_level != "High":
            risk_level = "Medium"
        risk_factors.append(f"Affects {len(critical_models)} critical models")
    
    metrics['risk_level'] = risk_level
    metrics['risk_factors'] = risk_factors
    metrics['critical_models'] = sorted(critical_models, key=lambda x: x[1], reverse=True)
    
    return metrics

def create_impact_tables(graph, selected_model, upstream_nodes, downstream_nodes):
    """Create tables of impacted models."""
    # Upstream table
    upstream_data = []
    for node_id in upstream_nodes:
        attrs = graph.nodes[node_id]
        node_type = attrs.get('node_type', 'unknown')
        name = attrs.get('name', node_id)
        
        # Get properties
        materialization = attrs.get('materialization', 'unknown')
        
        # Add to data
        upstream_data.append({
            'id': node_id,
            'name': name,
            'type': node_type,
            'materialization': materialization
        })
    
    # Downstream table
    downstream_data = []
    for node_id in downstream_nodes:
        attrs = graph.nodes[node_id]
        node_type = attrs.get('node_type', 'unknown')
        name = attrs.get('name', node_id)
        
        # Get properties
        materialization = attrs.get('materialization', 'unknown')
        
        # Add to data
        downstream_data.append({
            'id': node_id,
            'name': name,
            'type': node_type,
            'materialization': materialization
        })
    
    # Convert to dataframes
    upstream_df = pd.DataFrame(upstream_data)
    downstream_df = pd.DataFrame(downstream_data)
    
    return upstream_df, downstream_df

def main():
    """Main Streamlit app."""
    st.title("🔍 dbt Model Impact Analysis")
    st.markdown("""
    Analyze the impact of changes to a specific model by visualizing its upstream and downstream dependencies.
    This helps assess the risk and scope of changes before implementation.
    """)
    
    # Load graph
    graph = load_graph()
    if graph is None:
        st.error("Failed to load graph data. Please check the data files.")
        return
    
    # Initialize analyzer
    analyzer = GraphAnalyzer(graph)
    
    # Sidebar
    st.sidebar.title("Impact Analysis Settings")
    
    # Get model list
    models = get_model_list(graph)
    if not models:
        st.error("No models found in the graph. Please check your data.")
        return
        
    model_names = [name for _, name in models]
    model_ids = [id for id, _ in models]
    
    # Model selection
    selected_model_name = st.sidebar.selectbox("Select a model to analyze", model_names)
    selected_model_index = model_names.index(selected_model_name)
    selected_model = model_ids[selected_model_index]
    
    # Depth settings
    st.sidebar.subheader("Dependency Depth")
    upstream_depth = st.sidebar.slider("Upstream Depth", 1, 10, 3, 
                                      help="How many levels of parent dependencies to include")
    downstream_depth = st.sidebar.slider("Downstream Depth", 1, 10, 3,
                                        help="How many levels of child dependencies to include")
    
    # Include tests
    include_tests = st.sidebar.checkbox("Include Tests", True,
                                       help="Include test nodes in the visualization")
    
    # Create impact subgraph
    subgraph, upstream_nodes, downstream_nodes = create_impact_subgraph(
        graph, selected_model, upstream_depth, downstream_depth, include_tests
    )
    
    # Calculate impact metrics
    metrics = create_impact_metrics(graph, selected_model, upstream_nodes, downstream_nodes)
    
    # Main content - metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Upstream Dependencies", metrics['upstream_count'])
    
    with col2:
        st.metric("Downstream Impact", metrics['downstream_count'])
    
    with col3:
        st.metric("Total Affected Nodes", metrics['total_impact'])
    
    with col4:
        st.metric("Risk Level", metrics['risk_level'],
                 delta="High Risk" if metrics['risk_level'] == "High" else None)
    
    # Risk factors
    if metrics['risk_factors']:
        st.subheader("⚠️ Risk Factors")
        for factor in metrics['risk_factors']:
            st.warning(factor)
    
    # Critical models
    if metrics['critical_models']:
        st.subheader("🚨 Critical Downstream Models")
        critical_df = pd.DataFrame(metrics['critical_models'], columns=['Model', 'Downstream Dependencies'])
        st.dataframe(critical_df, use_container_width=True)
    
    # Visualization
    st.subheader("Impact Visualization")
    
    # Create visualization
    nodes, edges, config = create_impact_visualization(
        graph, subgraph, selected_model, upstream_nodes, downstream_nodes
    )
    
    if len(nodes) > 0:
        st.write(f"Showing impact graph with {len(nodes)} nodes and {len(edges)} edges")
        agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.warning("No dependencies to display with current settings.")
    
    # Detailed tables
    st.subheader("Detailed Impact Analysis")
    
    # Create tables
    upstream_df, downstream_df = create_impact_tables(graph, selected_model, upstream_nodes, downstream_nodes)
    
    # Display tables
    tab1, tab2 = st.tabs(["Upstream Dependencies", "Downstream Impact"])
    
    with tab1:
        if not upstream_df.empty:
            st.write(f"Showing {len(upstream_df)} upstream dependencies")
            st.dataframe(upstream_df, use_container_width=True)
        else:
            st.info("No upstream dependencies found.")
    
    with tab2:
        if not downstream_df.empty:
            st.write(f"Showing {len(downstream_df)} downstream dependencies")
            st.dataframe(downstream_df, use_container_width=True)
        else:
            st.info("No downstream dependencies found.")
    
    # Model details
    st.subheader("Selected Model Details")
    
    # Get model attributes
    model_attrs = graph.nodes[selected_model]
    model_name = model_attrs.get('name', selected_model)
    model_type = model_attrs.get('node_type', 'unknown')
    
    # Display model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Information**")
        st.write(f"ID: `{selected_model}`")
        st.write(f"Name: {model_name}")
        st.write(f"Type: {model_type}")
        
        if 'materialization' in model_attrs:
            st.write(f"Materialization: {model_attrs['materialization']}")
        
        if 'database_name' in model_attrs and 'schema_name' in model_attrs:
            st.write(f"Location: {model_attrs.get('database_name')}.{model_attrs.get('schema_name')}")
    
    with col2:
        st.write("**Additional Properties**")
        for k, v in model_attrs.items():
            if k not in ['name', 'node_type', 'materialization', 'database_name', 'schema_name'] and isinstance(v, (str, int, float, bool)) or v is None:
                st.write(f"{k}: {v}")
    
    # Action recommendations
    st.subheader("Recommended Actions")
    
    if metrics['risk_level'] == "High":
        st.error("""
        **High Risk Change - Proceed with Caution**
        - Create a dedicated test branch for this change
        - Run full regression tests before merging
        - Consider implementing the change incrementally
        - Schedule the deployment during off-hours
        - Have a rollback plan ready
        """)
    elif metrics['risk_level'] == "Medium":
        st.warning("""
        **Medium Risk Change - Take Precautions**
        - Test thoroughly before deploying
        - Notify downstream stakeholders
        - Monitor closely after deployment
        """)
    else:
        st.success("""
        **Low Risk Change - Standard Process**
        - Follow normal development workflow
        - Run standard tests
        - Deploy during regular deployment window
        """)


if __name__ == "__main__":
    main()