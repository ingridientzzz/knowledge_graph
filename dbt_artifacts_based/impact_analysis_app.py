#!/usr/bin/env python3
"""
Streamlit app for dbt impact analysis using manifest-parsed data.
Visualizes and analyzes the impact of changes through the knowledge graph.
"""

import streamlit as st
import pandas as pd
import networkx as nx
import json
import pickle
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from streamlit_agraph import agraph, Node, Edge, Config
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

# Set page config
st.set_page_config(
    page_title="dbt Impact Analysis - Manifest Based",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
DATA_DIR = Path("data")
VIZ_DIR = Path("visualization")

# Node type mappings for display
NODE_TYPE_COLORS = {
    "model": "#4287f5",          # blue
    "test": "#42f59e",           # green
    "source": "#f5a142",         # orange
    "column": "#f542e5",         # pink
    "exposure": "#9c42f5",       # purple
    "metric": "#f54242",         # red
    "package": "#42f5f5",        # cyan
    "macro": "#f5f542",          # yellow
    "seed": "#8cf542",           # light green
    "snapshot": "#f58c42",       # orange-red
    "analysis": "#8c42f5",       # violet
    "unknown": "#999999",        # gray
    "selected": "#ff0000",       # bright red for selected
    "upstream": "#ff9900",       # orange for upstream
    "downstream": "#0099ff",     # bright blue for downstream
}

NODE_TYPE_SHAPES = {
    "model": "dot",
    "test": "triangle",
    "source": "diamond",
    "column": "square",
    "exposure": "star",
    "metric": "hexagon",
    "package": "box",
    "macro": "ellipse",
    "seed": "database",
    "snapshot": "image",
    "analysis": "text",
}

@st.cache_data
def load_graph_data() -> Optional[nx.MultiDiGraph]:
    """Load graph data from stored files with caching."""
    try:
        # Try to load from pickle first (fastest)
        pickle_file = DATA_DIR / "knowledge_graph.gpickle"
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                graph = pickle.load(f)
            return graph
        
        # Fall back to JSON if pickle fails
        nodes_file = DATA_DIR / "nodes.json"
        edges_file = DATA_DIR / "edges.json"
        
        if nodes_file.exists() and edges_file.exists():
            st.info("Loading graph from JSON files...")
            with open(nodes_file, 'r', encoding='utf-8') as f:
                nodes_data = json.load(f)
            
            with open(edges_file, 'r', encoding='utf-8') as f:
                edges_data = json.load(f)
            
            # Create graph
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
        
        else:
            st.error("No graph data found. Please run the manifest parser first.")
            st.code("python manifest_parser.py path/to/manifest.json --storage-dir data")
            return None
            
    except Exception as e:
        st.error(f"Failed to load graph data: {e}")
        return None

@st.cache_data
def load_metadata() -> Optional[Dict]:
    """Load graph metadata."""
    try:
        metadata_file = DATA_DIR / "graph_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Failed to load metadata: {e}")
        return None

def get_nodes_by_type(graph: nx.MultiDiGraph, node_type: str) -> List[Tuple[str, str]]:
    """Get list of nodes of a specific type."""
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('node_type') == node_type:
            name = attrs.get('name', node_id)
            nodes.append((node_id, name))
    
    return sorted(nodes, key=lambda x: x[1])

def get_upstream_nodes(graph: nx.MultiDiGraph, node_id: str, depth: Optional[int] = None) -> Set[str]:
    """Get upstream dependencies of a node."""
    if depth == 0:
        return set()
    
    upstream = set()
    for predecessor in graph.predecessors(node_id):
        upstream.add(predecessor)
        if depth is None or depth > 1:
            upstream.update(get_upstream_nodes(graph, predecessor, None if depth is None else depth - 1))
    
    return upstream

def get_downstream_nodes(graph: nx.MultiDiGraph, node_id: str, depth: Optional[int] = None) -> Set[str]:
    """Get downstream dependencies of a node."""
    if depth == 0:
        return set()
    
    downstream = set()
    for successor in graph.successors(node_id):
        downstream.add(successor)
        if depth is None or depth > 1:
            downstream.update(get_downstream_nodes(graph, successor, None if depth is None else depth - 1))
    
    return downstream

def create_impact_subgraph(graph: nx.MultiDiGraph, selected_node: str, 
                          upstream_depth: Optional[int], downstream_depth: Optional[int],
                          include_tests: bool = True, include_columns: bool = False,
                          include_packages: bool = False) -> Tuple[nx.MultiDiGraph, Set[str], Set[str]]:
    """Create a subgraph showing the impact of the selected node."""
    upstream_nodes = get_upstream_nodes(graph, selected_node, upstream_depth)
    downstream_nodes = get_downstream_nodes(graph, selected_node, downstream_depth)
    
    # Create node set
    nodes_to_include = {selected_node} | upstream_nodes | downstream_nodes
    
    # Filter based on options
    if not include_tests:
        nodes_to_include = {
            node_id for node_id in nodes_to_include
            if graph.nodes[node_id].get('node_type') != 'test'
        }
    
    if not include_columns:
        nodes_to_include = {
            node_id for node_id in nodes_to_include
            if graph.nodes[node_id].get('node_type') != 'column'
        }
    
    if not include_packages:
        nodes_to_include = {
            node_id for node_id in nodes_to_include
            if graph.nodes[node_id].get('node_type') != 'package'
        }
    
    # Create subgraph
    subgraph = graph.subgraph(nodes_to_include)
    
    return subgraph, upstream_nodes, downstream_nodes

def create_impact_visualization(graph: nx.MultiDiGraph, subgraph: nx.MultiDiGraph, 
                               selected_node: str, upstream_nodes: Set[str], 
                               downstream_nodes: Set[str]) -> Tuple[List[Node], List[Edge], Config]:
    """Create interactive visualization of the impact graph."""
    nodes = []
    edges = []
    
    # Create nodes
    for node_id, attrs in subgraph.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        name = attrs.get('name', node_id)
        
        # Determine node color and size based on impact
        if node_id == selected_node:
            color = NODE_TYPE_COLORS['selected']
            size = 30
        elif node_id in upstream_nodes:
            color = NODE_TYPE_COLORS['upstream']
            size = 20
        elif node_id in downstream_nodes:
            color = NODE_TYPE_COLORS['downstream']
            size = 20
        else:
            color = NODE_TYPE_COLORS.get(node_type, NODE_TYPE_COLORS['unknown'])
            size = 15
        
        # Create tooltip
        tooltip = f"<b>{name}</b><br>Type: {node_type}<br>ID: {node_id}"
        if 'description' in attrs and attrs['description']:
            desc = attrs['description'][:100] + "..." if len(attrs['description']) > 100 else attrs['description']
            tooltip += f"<br>Description: {desc}"
        if 'package_name' in attrs:
            tooltip += f"<br>Package: {attrs['package_name']}"
        
        # Get shape
        shape = NODE_TYPE_SHAPES.get(node_type, "dot")
        
        nodes.append(Node(
            id=node_id,
            label=name,
            size=size,
            color=color,
            shape=shape,
            title=tooltip
        ))
    
    # Create edges
    for source, target, attrs in subgraph.edges(data=True):
        edge_type = attrs.get('edge_type', 'related')
        
        # Determine edge styling based on impact
        if source == selected_node or target == selected_node:
            color = "#ff0000"
            width = 3
        elif source in upstream_nodes and target == selected_node:
            color = NODE_TYPE_COLORS['upstream']
            width = 2
        elif source == selected_node and target in downstream_nodes:
            color = NODE_TYPE_COLORS['downstream']
            width = 2
        else:
            color = "#666666"
            width = 1
        
        edges.append(Edge(
            source=source,
            target=target,
            label=edge_type,
            title=f"Relationship: {edge_type}",
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

def calculate_impact_metrics(graph: nx.MultiDiGraph, selected_node: str, 
                           upstream_nodes: Set[str], downstream_nodes: Set[str]) -> Dict[str, Any]:
    """Calculate comprehensive impact metrics."""
    metrics = {}
    
    # Basic counts
    metrics['upstream_count'] = len(upstream_nodes)
    metrics['downstream_count'] = len(downstream_nodes)
    metrics['total_impact'] = len(upstream_nodes) + len(downstream_nodes) + 1
    
    # Node type breakdown
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
    
    # Package impact
    upstream_packages = set()
    downstream_packages = set()
    
    for node_id in upstream_nodes:
        package = graph.nodes[node_id].get('package_name')
        if package:
            upstream_packages.add(package)
    
    for node_id in downstream_nodes:
        package = graph.nodes[node_id].get('package_name')
        if package:
            downstream_packages.add(package)
    
    metrics['upstream_packages'] = list(upstream_packages)
    metrics['downstream_packages'] = list(downstream_packages)
    metrics['total_packages_affected'] = len(upstream_packages | downstream_packages)
    
    # Risk assessment
    risk_level = "Low"
    risk_factors = []
    
    if len(downstream_nodes) > 20:
        risk_level = "High"
        risk_factors.append(f"Very large downstream impact ({len(downstream_nodes)} nodes)")
    elif len(downstream_nodes) > 10:
        risk_level = "Medium"
        risk_factors.append(f"Large downstream impact ({len(downstream_nodes)} nodes)")
    elif len(downstream_nodes) > 5:
        risk_level = "Medium"
        risk_factors.append(f"Moderate downstream impact ({len(downstream_nodes)} nodes)")
    
    # Check for critical downstream nodes
    critical_nodes = []
    for node_id in downstream_nodes:
        downstream_count = len(list(graph.successors(node_id)))
        if downstream_count > 10:
            name = graph.nodes[node_id].get('name', node_id)
            node_type = graph.nodes[node_id].get('node_type', 'unknown')
            critical_nodes.append((name, downstream_count, node_type))
    
    if critical_nodes:
        if risk_level == "Low":
            risk_level = "Medium"
        risk_factors.append(f"Affects {len(critical_nodes)} highly connected nodes")
    
    # Package diversity risk
    if len(downstream_packages) > 5:
        if risk_level != "High":
            risk_level = "Medium"
        risk_factors.append(f"Affects {len(downstream_packages)} different packages")
    
    metrics['risk_level'] = risk_level
    metrics['risk_factors'] = risk_factors
    metrics['critical_nodes'] = sorted(critical_nodes, key=lambda x: x[1], reverse=True)
    
    return metrics

def _create_node_data(graph: nx.MultiDiGraph, node_ids: Set[str]) -> List[Dict]:
    """Helper function to create node data for impact tables."""
    data = []
    for node_id in node_ids:
        attrs = graph.nodes[node_id]
        row = {
            'name': attrs.get('name', node_id),
            'type': attrs.get('node_type', 'unknown'),
            'dbt_project': attrs.get('package_name', 'unknown'),
            'description': attrs.get('description', '')[:100] + "..." if attrs.get('description', '') and len(attrs.get('description', '')) > 100 else attrs.get('description', ''),
            'id': node_id,
        }
        
        # Add type-specific fields
        if attrs.get('node_type') == 'model':
            row['materialization'] = attrs.get('materialization', '')
            row['database'] = attrs.get('database', '')
            row['schema'] = attrs.get('schema', '')
        elif attrs.get('node_type') == 'source':
            row['source_name'] = attrs.get('source_name', '')
            row['database'] = attrs.get('database', '')
            row['schema'] = attrs.get('schema', '')
        
        data.append(row)
    return data

def create_impact_tables(graph: nx.MultiDiGraph, upstream_nodes: Set[str], 
                        downstream_nodes: Set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create detailed tables of impacted nodes."""
    upstream_data = _create_node_data(graph, upstream_nodes)
    downstream_data = _create_node_data(graph, downstream_nodes)
    
    upstream_df = pd.DataFrame(upstream_data)
    downstream_df = pd.DataFrame(downstream_data)
    
    return upstream_df, downstream_df

def create_package_impact_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create a chart showing package impact distribution."""
    all_packages = set(metrics['upstream_packages']) | set(metrics['downstream_packages'])
    
    package_data = []
    for package in all_packages:
        upstream_count = 1 if package in metrics['upstream_packages'] else 0
        downstream_count = 1 if package in metrics['downstream_packages'] else 0
        package_data.append({
            'package': package,
            'upstream': upstream_count,
            'downstream': downstream_count,
            'total': upstream_count + downstream_count
        })
    
    df = pd.DataFrame(package_data)
    
    if not df.empty:
        fig = px.bar(df, x='package', y=['upstream', 'downstream'], 
                     title="Package Impact Distribution",
                     labels={'value': 'Impact Type', 'package': 'Package Name'},
                     color_discrete_map={'upstream': '#ff9900', 'downstream': '#0099ff'})
        fig.update_layout(height=400)
        return fig
    
    return go.Figure()

def main():
    """Main Streamlit app."""
    st.title("🔍 dbt Impact Analysis - Manifest Based")
    st.markdown("""
    Analyze the impact of changes to dbt resources using the knowledge graph built from manifest.json.
    Visualize dependencies, assess risk, and understand the scope of changes.
    """)
    
    # Load data
    with st.spinner("Loading graph data..."):
        graph = load_graph_data()
        metadata = load_metadata()
    
    if graph is None:
        st.error("Failed to load graph data. Please ensure the manifest parser has been run.")
        st.info("Run: `python manifest_parser.py path/to/manifest.json --storage-dir data`")
        return
    
    # Sidebar
    st.sidebar.title("Impact Analysis Configuration")
    
    # Display metadata
    if metadata:
        st.sidebar.subheader("📊 Graph Statistics")
        stats = metadata.get('statistics', {})
        st.sidebar.metric("Total Nodes", stats.get('total_nodes', 0))
        st.sidebar.metric("Total Edges", stats.get('total_edges', 0))
        st.sidebar.metric("dbt Version", stats.get('manifest_version', 'unknown'))
    
    # Node selection
    st.sidebar.subheader("🎯 Node Selection")
    
    available_types = ['model', 'source', 'test', 'exposure', 'metric', 'seed', 'snapshot', 'analysis']
    node_type = st.sidebar.selectbox(
        "Select node type",
        available_types,
        help="Choose the type of dbt resource to analyze"
    )
    
    # Get nodes of selected type
    nodes_of_type = get_nodes_by_type(graph, node_type)
    
    if not nodes_of_type:
        st.error(f"No {node_type} nodes found in the graph.")
        return
    
    # Search/filter nodes
    search_term = st.sidebar.text_input(
        f"Search {node_type} nodes",
        help="Search by name or ID"
    )
    
    if search_term:
        filtered_nodes = [
            (node_id, name) for node_id, name in nodes_of_type
            if search_term.lower() in name.lower() or search_term.lower() in node_id.lower()
        ]
    else:
        filtered_nodes = nodes_of_type
    
    if not filtered_nodes:
        st.warning(f"No {node_type} nodes match the search term.")
        return
    
    # Node selection
    node_names = [name for _, name in filtered_nodes]
    node_ids = [node_id for node_id, _ in filtered_nodes]
    
    selected_name = st.sidebar.selectbox(f"Select {node_type}", node_names)
    selected_node = node_ids[node_names.index(selected_name)]
    
    # Analysis options
    st.sidebar.subheader("🔧 Analysis Options")
    
    upstream_depth = st.sidebar.slider(
        "Upstream Depth", 1, 10, 3,
        help="How many levels of upstream dependencies to include"
    )
    
    downstream_depth = st.sidebar.slider(
        "Downstream Depth", 1, 10, 3,
        help="How many levels of downstream dependencies to include"
    )
    
    include_tests = st.sidebar.checkbox("Include Tests", True)
    include_columns = st.sidebar.checkbox("Include Columns", False)
    include_packages = st.sidebar.checkbox("Include Packages", False)
    
    # Perform impact analysis
    with st.spinner("Analyzing impact..."):
        subgraph, upstream_nodes, downstream_nodes = create_impact_subgraph(
            graph, selected_node, upstream_depth, downstream_depth,
            include_tests, include_columns, include_packages
        )
        
        metrics = calculate_impact_metrics(graph, selected_node, upstream_nodes, downstream_nodes)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Upstream Dependencies", metrics['upstream_count'])
    
    with col2:
        st.metric("Downstream Impact", metrics['downstream_count'])
    
    with col3:
        st.metric("Total Affected Nodes", metrics['total_impact'])
    
    with col4:
        risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        st.metric(
            "Risk Level", 
            f"{risk_color.get(metrics['risk_level'], '⚪')} {metrics['risk_level']}"
        )
    
    # Risk factors
    if metrics['risk_factors']:
        st.subheader("⚠️ Risk Factors")
        for factor in metrics['risk_factors']:
            st.warning(factor)
    
    # Package impact
    if metrics['total_packages_affected'] > 0:
        st.subheader("📦 Package Impact")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Packages Affected", metrics['total_packages_affected'])
            if metrics['upstream_packages']:
                st.write("**Upstream Packages:**")
                for pkg in metrics['upstream_packages']:
                    st.write(f"• {pkg}")
        
        with col2:
            if metrics['downstream_packages']:
                st.write("**Downstream Packages:**")
                for pkg in metrics['downstream_packages']:
                    st.write(f"• {pkg}")
        
        # Package impact chart
        if len(metrics['upstream_packages']) + len(metrics['downstream_packages']) > 0:
            fig = create_package_impact_chart(metrics)
            if fig.data:
                st.plotly_chart(fig, use_container_width=True)
    
    # Critical nodes
    if metrics['critical_nodes']:
        st.subheader("🚨 Critical Downstream Nodes")
        critical_df = pd.DataFrame(
            metrics['critical_nodes'],
            columns=['Name', 'Downstream Count', 'Type']
        )
        st.dataframe(critical_df, use_container_width=True)
    
    # Visualization
    st.subheader("🎨 Impact Visualization")
    
    if len(subgraph.nodes()) > 0:
        nodes, edges, config = create_impact_visualization(
            graph, subgraph, selected_node, upstream_nodes, downstream_nodes
        )
        
        st.write(f"Showing impact graph with {len(nodes)} nodes and {len(edges)} edges")
        
        # Legend
        with st.expander("🎯 Visualization Legend"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Node Colors:**")
                st.write("🔴 Selected Node")
                st.write("🟠 Upstream Dependencies")
                st.write("🔵 Downstream Impact")
            with col2:
                st.write("**Node Shapes:**")
                st.write("● Models")
                st.write("▲ Tests")
                st.write("♦ Sources")
                st.write("■ Columns")
                st.write("⭐ Exposures")
        
        agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.warning("No dependencies to display with current settings.")
    
    # Detailed analysis
    st.subheader("📋 Detailed Impact Analysis")
    
    upstream_df, downstream_df = create_impact_tables(graph, upstream_nodes, downstream_nodes)
    
    tab1, tab2, tab3 = st.tabs(["Upstream Dependencies", "Downstream Impact", "Node Details"])
    
    with tab1:
        if not upstream_df.empty:
            st.write(f"**{len(upstream_df)} upstream dependencies**")
            st.dataframe(upstream_df, use_container_width=True)
            
            # Type breakdown
            if len(upstream_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    type_counts = upstream_df['type'].value_counts()
                    fig = px.pie(values=type_counts.values, names=type_counts.index, 
                               title="Upstream Node Types")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'dbt_project' in upstream_df.columns:
                        package_counts = upstream_df['dbt_project'].value_counts()
                        fig = px.pie(values=package_counts.values, names=package_counts.index, 
                                   title="Upstream dbt Projects")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No upstream dependencies found.")
    
    with tab2:
        if not downstream_df.empty:
            st.write(f"**{len(downstream_df)} downstream dependencies**")
            st.dataframe(downstream_df, use_container_width=True)
            
            # Type breakdown
            if len(downstream_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    type_counts = downstream_df['type'].value_counts()
                    fig = px.pie(values=type_counts.values, names=type_counts.index, 
                               title="Downstream Node Types")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'dbt_project' in downstream_df.columns:
                        package_counts = downstream_df['dbt_project'].value_counts()
                        fig = px.pie(values=package_counts.values, names=package_counts.index, 
                                   title="Downstream dbt Projects")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No downstream dependencies found.")
    
    with tab3:
        st.subheader(f"Selected {node_type.title()} Details")
        
        # Get node attributes
        node_attrs = graph.nodes[selected_node]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information**")
            st.write(f"**ID:** `{selected_node}`")
            st.write(f"**Name:** {node_attrs.get('name', 'N/A')}")
            st.write(f"**Type:** {node_attrs.get('node_type', 'N/A')}")
            
            if node_attrs.get('package_name'):
                st.write(f"**Package:** {node_attrs['package_name']}")
            
            if node_attrs.get('description'):
                st.write(f"**Description:** {node_attrs['description']}")
        
        with col2:
            st.write("**Additional Properties**")
            
            # Type-specific properties
            if node_type == 'model':
                if node_attrs.get('materialization'):
                    st.write(f"**Materialization:** {node_attrs['materialization']}")
                if node_attrs.get('database') and node_attrs.get('schema'):
                    st.write(f"**Location:** {node_attrs['database']}.{node_attrs['schema']}")
            
            elif node_type == 'source':
                if node_attrs.get('source_name'):
                    st.write(f"**Source Name:** {node_attrs['source_name']}")
                if node_attrs.get('database') and node_attrs.get('schema'):
                    st.write(f"**Location:** {node_attrs['database']}.{node_attrs['schema']}")
            
            # Tags
            if node_attrs.get('tags'):
                tags = node_attrs['tags']
                if isinstance(tags, list) and tags:
                    st.write(f"**Tags:** {', '.join(tags)}")
    
    # Action recommendations
    st.subheader("💡 Recommended Actions")
    
    if metrics['risk_level'] == "High":
        st.error("""
        **High Risk Change - Proceed with Extreme Caution**
        - Create a dedicated development branch
        - Run comprehensive impact testing
        - Coordinate with all affected package owners
        - Schedule deployment during maintenance window
        - Prepare detailed rollback plan
        - Consider phased deployment approach
        """)
    elif metrics['risk_level'] == "Medium":
        st.warning("""
        **Medium Risk Change - Take Precautions**
        - Test thoroughly in development environment
        - Notify stakeholders of affected downstream resources
        - Run regression tests on critical downstream models
        - Monitor closely after deployment
        """)
    else:
        st.success("""
        **Low Risk Change - Standard Process**
        - Follow normal development workflow
        - Run standard test suite
        - Deploy during regular deployment window
        """)


if __name__ == "__main__":
    main()