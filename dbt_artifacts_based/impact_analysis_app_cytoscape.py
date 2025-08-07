#!/usr/bin/env python3
"""
Streamlit app for dbt impact analysis using manifest-parsed data.
Cytoscape.js-based version using st-link-analysis for Snowflake compatibility.
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
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

# st-link-analysis imports
try:
    from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle
    CYTOSCAPE_AVAILABLE = True
except ImportError:
    CYTOSCAPE_AVAILABLE = False
    st.error("st-link-analysis is not installed. Please install it with: pip install st-link-analysis")

# Set page config
st.set_page_config(
    page_title="dbt Impact Analysis - Cytoscape Version",
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
    "model": "ellipse",
    "test": "rectangle", 
    "source": "diamond",
    "column": "triangle",
    "exposure": "star",
    "metric": "hexagon",
    "package": "octagon",
    "macro": "pentagon",
    "seed": "round-rectangle",
    "snapshot": "round-triangle",
    "analysis": "parallelogram",
}

# Cytoscape layout algorithms
LAYOUT_ALGORITHMS = {
    "dagre": "Hierarchical (DAG-based)",
    "breadthfirst": "Breadth-first Tree",
    "circle": "Circular Layout",
    "concentric": "Concentric Circles",
    "grid": "Grid Layout",
    "random": "Random Positions",
    "cose": "Compound Spring Embedder",
    "fcose": "Fast Compound Spring Embedder",
    "cola": "Constraint-based Layout"
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
                    attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
                    graph.add_edge(source, target, **attrs)
            
            return graph
        
        else:
            st.error("No graph data found. Please generate graph data first.")
            st.info("Run: `python manifest_parser.py path/to/manifest.json --storage-dir data`")
            return None
            
    except Exception as e:
        st.error(f"Error loading graph data: {e}")
        return None

@st.cache_data
def load_metadata() -> Optional[Dict]:
    """Load graph metadata if available."""
    try:
        metadata_file = DATA_DIR / "graph_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load metadata: {e}")
    return None

def get_upstream_nodes(graph: nx.MultiDiGraph, start_node: str, depth: int) -> Set[str]:
    """Get upstream nodes up to specified depth."""
    if depth <= 0:
        return set()
    
    upstream = set()
    current_level = {start_node}
    
    for _ in range(depth):
        next_level = set()
        for node in current_level:
            predecessors = set(graph.predecessors(node))
            next_level.update(predecessors)
            upstream.update(predecessors)
        current_level = next_level
        if not current_level:
            break
    
    return upstream

def get_downstream_nodes(graph: nx.MultiDiGraph, start_node: str, depth: int) -> Set[str]:
    """Get downstream nodes up to specified depth."""
    if depth <= 0:
        return set()
    
    downstream = set()
    current_level = {start_node}
    
    for _ in range(depth):
        next_level = set()
        for node in current_level:
            successors = set(graph.successors(node))
            next_level.update(successors)
            downstream.update(successors)
        current_level = next_level
        if not current_level:
            break
    
    return downstream

def get_nodes_by_type(graph: nx.MultiDiGraph, node_type: str) -> List[Tuple[str, str]]:
    """Get list of nodes of a specific type with safety checks."""
    if not graph or len(graph.nodes) == 0:
        return []
        
    if node_type == 'source':
        return _get_deduplicated_sources(graph)
    
    nodes = []
    try:
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get('node_type') == node_type:
                base_name = attrs.get('name', node_id)
                package_name = attrs.get('package_name', '')
                if package_name:
                    display_name = f"{base_name} ({package_name})"
                else:
                    display_name = base_name
                nodes.append((node_id, display_name))
    except Exception as e:
        st.error(f"Error reading {node_type} nodes: {e}")
        return []
    
    return sorted(nodes, key=lambda x: x[1])

def _get_deduplicated_sources(graph: nx.MultiDiGraph) -> List[Tuple[str, str]]:
    """Get deduplicated source nodes for dropdown selection."""
    source_groups = {}
    
    # Group source nodes by their actual source name
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('node_type') == 'source':
            source_name = attrs.get('name', node_id)
            
            if source_name not in source_groups:
                source_groups[source_name] = {
                    'representative_id': node_id,
                    'display_name': source_name,
                    'count': 0
                }
            source_groups[source_name]['count'] += 1
    
    # Return list of (representative_id, display_name) for dropdown
    deduplicated = []
    for source_info in source_groups.values():
        deduplicated.append((source_info['representative_id'], source_info['display_name']))
    
    return sorted(deduplicated, key=lambda x: x[1])

def create_impact_subgraph(graph: nx.MultiDiGraph, selected_node: str,
                          upstream_depth: Optional[int], downstream_depth: Optional[int],
                          include_tests: bool = True, include_columns: bool = False,
                          include_packages: bool = False) -> Tuple[nx.MultiDiGraph, Set[str], Set[str]]:
    """Create a subgraph showing the impact of the selected node."""
    
    # For sources, aggregate impact from all nodes with the same source name
    if selected_node not in graph.nodes:
        st.error(f"Selected node '{selected_node}' not found in graph. Please regenerate the graph data.")
        return None
        
    if graph.nodes[selected_node].get('node_type') == 'source':
        return _create_aggregated_source_impact(graph, selected_node, upstream_depth, downstream_depth,
                                              include_tests, include_columns, include_packages)
    
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

def _create_aggregated_source_impact(graph: nx.MultiDiGraph, selected_node: str,
                                   upstream_depth: Optional[int], downstream_depth: Optional[int],
                                   include_tests: bool, include_columns: bool, include_packages: bool) -> Tuple[nx.MultiDiGraph, Set[str], Set[str]]:
    """Create impact analysis for a source by aggregating all source nodes with the same name."""
    
    # Get the source name from the selected node
    if selected_node not in graph.nodes:
        return None, set(), set()
        
    selected_source_name = graph.nodes[selected_node].get('name', '')
    
    # Find all source nodes with the same name across different packages
    all_related_sources = []
    for node_id, attrs in graph.nodes(data=True):
        if (attrs.get('node_type') == 'source' and 
            attrs.get('name', '') == selected_source_name):
            all_related_sources.append(node_id)
    
    # Aggregate upstream and downstream from all related sources
    all_upstream = set()
    all_downstream = set()
    
    for source_node in all_related_sources:
        upstream = get_upstream_nodes(graph, source_node, upstream_depth)
        downstream = get_downstream_nodes(graph, source_node, downstream_depth)
        all_upstream.update(upstream)
        all_downstream.update(downstream)
    
    # Create node set including all related sources
    nodes_to_include = set(all_related_sources) | all_upstream | all_downstream
    
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
    
    return subgraph, all_upstream, all_downstream

def create_cytoscape_visualization(graph: nx.MultiDiGraph, subgraph: nx.MultiDiGraph,
                                 selected_node: str, upstream_nodes: Set[str], 
                                 downstream_nodes: Set[str], layout_algorithm: str = "dagre") -> Dict:
    """Create interactive Cytoscape visualization and return the result."""
    
    if not CYTOSCAPE_AVAILABLE:
        return None
    
    # First, consolidate source nodes with the same name
    consolidated_nodes, node_mapping = _consolidate_source_nodes_for_viz(graph, subgraph, selected_node)
    
    # Prepare elements for Cytoscape (nodes and edges separate)
    nodes = []
    edges = []
    
    # Add nodes
    for consolidated_id, node_info in consolidated_nodes.items():
        node_type = node_info['node_type']
        name = node_info['name']
        
        # Determine node color and size based on impact
        if selected_node in node_info['original_ids']:
            color = NODE_TYPE_COLORS['selected']
            size = 60
            border_width = 4
        elif any(node_id in upstream_nodes for node_id in node_info['original_ids']):
            color = NODE_TYPE_COLORS['upstream']
            size = 45
            border_width = 3
        elif any(node_id in downstream_nodes for node_id in node_info['original_ids']):
            color = NODE_TYPE_COLORS['downstream']
            size = 45
            border_width = 3
        else:
            color = NODE_TYPE_COLORS.get(node_type, NODE_TYPE_COLORS['unknown'])
            size = 35
            border_width = 2
        
        # Get shape
        shape = NODE_TYPE_SHAPES.get(node_type, "ellipse")
        
        # Create label
        label = name[:20] + "..." if len(name) > 20 else name
        if node_info['is_consolidated']:
            label += f"\n({len(node_info['original_ids'])} pkgs)"
        
        # Create node element for Cytoscape (st-link-analysis format)
        node_element = {
            "data": {
                "id": consolidated_id,
                "label": node_type,  # This is used for styling by NodeStyle
                "caption": label,    # This is the display text
                "name": name,
                "type": node_type,
                "consolidated": node_info['is_consolidated'],
                "package_count": len(node_info['original_ids']) if node_info['is_consolidated'] else 1
            }
        }
        nodes.append(node_element)
    
    # Add edges
    edge_elements = _create_consolidated_edges_cytoscape(subgraph, node_mapping, consolidated_nodes, 
                                                       selected_node, upstream_nodes, downstream_nodes)
    edges.extend(edge_elements)
    
    # Combine nodes and edges into elements list (st-link-analysis format)
    elements = nodes + edges
    
    # Configure layout (st-link-analysis expects just the name or config dict)
    if layout_algorithm == "dagre":
        layout_config = {
            "name": "dagre",
            "rankDir": "TB",  # Top to bottom
            "nodeSep": 50,
            "rankSep": 100
        }
    elif layout_algorithm == "cose":
        layout_config = {
            "name": "cose",
            "nodeRepulsion": 10000,
            "nodeOverlap": 10,
            "idealEdgeLength": 100
        }
    elif layout_algorithm == "fcose":
        layout_config = {
            "name": "fcose",
            "nodeRepulsion": 10000,
            "idealEdgeLength": 100
        }
    elif layout_algorithm == "breadthfirst":
        layout_config = {
            "name": "breadthfirst",
            "directed": True,
            "roots": [node["data"]["id"] for node in nodes if selected_node in node["data"]["id"]][:1]
        }
    else:
        layout_config = layout_algorithm  # For simple string layouts
    
    return {
        "elements": elements,
        "layout": layout_config
    }

def _consolidate_source_nodes_for_viz(graph: nx.MultiDiGraph, subgraph: nx.MultiDiGraph, 
                                     selected_node: str) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """Consolidate source nodes with the same name for visualization."""
    
    # Group nodes by (node_type, name) for sources, keep others as-is
    node_groups = {}
    node_mapping = {}  # original_id -> consolidated_id
    
    for node_id, attrs in subgraph.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        name = attrs.get('name', node_id)
        
        if node_type == 'source':
            # Group sources by name
            group_key = f"source_{name}"
            
            if group_key not in node_groups:
                node_groups[group_key] = {
                    'name': name,
                    'node_type': node_type,
                    'original_ids': [],
                    'packages': set(),
                    'is_consolidated': False
                }
            
            node_groups[group_key]['original_ids'].append(node_id)
            if attrs.get('package_name'):
                node_groups[group_key]['packages'].add(attrs['package_name'])
            
            node_mapping[node_id] = group_key
        else:
            # Keep non-source nodes as individual nodes
            group_key = node_id
            node_groups[group_key] = {
                'name': name,
                'node_type': node_type,
                'original_ids': [node_id],
                'packages': {attrs.get('package_name', 'unknown')} if attrs.get('package_name') else set(),
                'is_consolidated': False
            }
            node_mapping[node_id] = group_key
    
    # Mark consolidated source groups
    for group_key, group_info in node_groups.items():
        if group_info['node_type'] == 'source' and len(group_info['original_ids']) > 1:
            group_info['is_consolidated'] = True
    
    return node_groups, node_mapping

def _create_consolidated_edges_cytoscape(subgraph: nx.MultiDiGraph, node_mapping: Dict[str, str], 
                                       consolidated_nodes: Dict[str, Dict], selected_node: str,
                                       upstream_nodes: Set[str], downstream_nodes: Set[str]) -> List[Dict]:
    """Create edges for the consolidated Cytoscape visualization."""
    
    edges = []
    edge_set = set()  # To avoid duplicate edges
    
    for source, target, attrs in subgraph.edges(data=True):
        # Safety check: ensure both source and target are in node_mapping
        if source not in node_mapping or target not in node_mapping:
            continue
            
        consolidated_source = node_mapping[source]
        consolidated_target = node_mapping[target]
        
        # Skip self-loops that might occur from consolidation
        if consolidated_source == consolidated_target:
            continue
        
        # Create unique edge identifier
        edge_key = (consolidated_source, consolidated_target)
        if edge_key in edge_set:
            continue
        edge_set.add(edge_key)
        
        # Safety check for consolidated nodes
        if consolidated_source not in consolidated_nodes:
            continue
        if consolidated_target not in consolidated_nodes:
            continue
            
        # Determine edge styling based on impact
        source_is_selected = selected_node in consolidated_nodes[consolidated_source]['original_ids']
        target_is_selected = selected_node in consolidated_nodes[consolidated_target]['original_ids']
        source_is_upstream = any(node_id in upstream_nodes for node_id in consolidated_nodes[consolidated_source]['original_ids'])
        target_is_downstream = any(node_id in downstream_nodes for node_id in consolidated_nodes[consolidated_target]['original_ids'])
        
        if source_is_selected or target_is_selected:
            color = "#ff0000"
            width = 4
            opacity = 1.0
        elif source_is_upstream or target_is_downstream:
            color = NODE_TYPE_COLORS.get('upstream', '#ff9900')
            width = 3
            opacity = 0.8
        else:
            color = "#666666"
            width = 2
            opacity = 0.6
        
        # Create edge element for Cytoscape (st-link-analysis format)
        edge_element = {
            "data": {
                "id": f"edge_{consolidated_source}_{consolidated_target}",
                "source": consolidated_source,
                "target": consolidated_target,
                "label": attrs.get('edge_type', 'depends_on'),
                "relationship": attrs.get('edge_type', 'depends_on')
            }
        }
        edges.append(edge_element)
    
    return edges

def calculate_impact_metrics(graph: nx.MultiDiGraph, selected_node: str, 
                           upstream_nodes: Set[str], downstream_nodes: Set[str]) -> Dict[str, Any]:
    """Calculate impact metrics for the selected node."""
    
    # Count metrics
    upstream_count = len(upstream_nodes)
    downstream_count = len(downstream_nodes)
    total_impact = upstream_count + downstream_count + 1  # +1 for selected node
    
    # Node type analysis
    upstream_by_type = {}
    downstream_by_type = {}
    
    for node_id in upstream_nodes:
        node_type = graph.nodes[node_id].get('node_type', 'unknown')
        upstream_by_type[node_type] = upstream_by_type.get(node_type, 0) + 1
    
    for node_id in downstream_nodes:
        node_type = graph.nodes[node_id].get('node_type', 'unknown')
        downstream_by_type[node_type] = downstream_by_type.get(node_type, 0) + 1
    
    # Package analysis
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
    
    total_packages_affected = len(upstream_packages | downstream_packages)
    
    # Risk assessment
    risk_level = "Low"
    risk_factors = []
    
    if downstream_count > 50:
        risk_level = "High"
        risk_factors.append(f"High downstream impact: {downstream_count} nodes affected")
    elif downstream_count > 20:
        risk_level = "Medium"
        risk_factors.append(f"Moderate downstream impact: {downstream_count} nodes affected")
    
    if total_packages_affected > 5:
        risk_level = "High"
        risk_factors.append(f"Cross-package impact: {total_packages_affected} packages affected")
    elif total_packages_affected > 2:
        if risk_level == "Low":
            risk_level = "Medium"
        risk_factors.append(f"Multi-package impact: {total_packages_affected} packages affected")
    
    # Find critical downstream nodes (high fan-out)
    critical_nodes = []
    for node_id in downstream_nodes:
        downstream_count = len(list(graph.successors(node_id)))
        if downstream_count > 10:
            name = graph.nodes[node_id].get('name', node_id)
            node_type = graph.nodes[node_id].get('node_type', 'unknown')
            critical_nodes.append((name, downstream_count, node_type))
    
    if critical_nodes:
        risk_factors.append(f"{len(critical_nodes)} critical downstream nodes with high fan-out")
        if risk_level == "Low":
            risk_level = "Medium"
    
    return {
        'upstream_count': upstream_count,
        'downstream_count': downstream_count,
        'total_impact': total_impact,
        'upstream_by_type': upstream_by_type,
        'downstream_by_type': downstream_by_type,
        'upstream_packages': upstream_packages,
        'downstream_packages': downstream_packages,
        'total_packages_affected': total_packages_affected,
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'critical_nodes': critical_nodes
    }

def _create_node_data(graph: nx.MultiDiGraph, node_ids: List[str], selected_node_type: str = None, aggregate_by_type: bool = False) -> List[Dict]:
    """Create node data for display in tables."""
    
    if aggregate_by_type:
        return _create_aggregated_node_data(graph, node_ids, selected_node_type)
    
    data = []
    for node_id in node_ids:
        if node_id not in graph.nodes:
            continue
            
        attrs = graph.nodes[node_id]
        row = {
            'name': attrs.get('name', node_id),
            'type': attrs.get('node_type', 'unknown'),
            'dbt_project': attrs.get('package_name', 'unknown'),
            'description': attrs.get('description', '')[:100] + "..." if attrs.get('description', '') and len(attrs.get('description', '')) > 100 else attrs.get('description', ''),
            'id': node_id,
        }
        
        # Add other relevant attributes
        for key in ['database', 'schema', 'alias', 'materialization', 'tags']:
            if key in attrs:
                row[key] = attrs[key]
        
        data.append(row)
    
    return data

def _create_aggregated_node_data(graph: nx.MultiDiGraph, node_ids: List[str], selected_node_type: str = None) -> List[Dict]:
    """Create aggregated node data grouped by type and package."""
    
    # Group nodes by type and package
    groups = {}
    
    for node_id in node_ids:
        if node_id not in graph.nodes:
            continue
            
        attrs = graph.nodes[node_id]
        node_type = attrs.get('node_type', 'unknown')
        package = attrs.get('package_name', 'unknown')
        
        # Filter out column nodes for source analysis
        if selected_node_type == 'source' and node_type == 'column':
            continue
        
        group_key = (node_type, package)
        
        if group_key not in groups:
            groups[group_key] = {
                'type': node_type,
                'dbt_project': package,
                'count': 0,
                'nodes': []
            }
        
        groups[group_key]['count'] += 1
        groups[group_key]['nodes'].append(attrs.get('name', node_id))
    
    # Convert to list format
    data = []
    for (node_type, package), group_info in groups.items():
        # Create a representative name
        if group_info['count'] == 1:
            name = group_info['nodes'][0]
        else:
            name = f"{node_type} nodes ({group_info['count']})"
        
        row = {
            'name': name,
            'type': node_type,
            'dbt_project': package,
            'count': group_info['count'],
            'description': f"{group_info['count']} {node_type} nodes in {package}",
            'id': f"{node_type}_{package}"
        }
        data.append(row)
    
    return data

def create_impact_tables(graph: nx.MultiDiGraph, upstream_nodes: Set[str], downstream_nodes: Set[str], 
                        selected_node_type: str = None, aggregate_view: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create tables showing upstream and downstream impact."""
    
    # Create upstream data
    upstream_data = _create_node_data(graph, list(upstream_nodes), selected_node_type, aggregate_view)
    upstream_df = pd.DataFrame(upstream_data)
    
    # Create downstream data
    downstream_data = _create_node_data(graph, list(downstream_nodes), selected_node_type, aggregate_view)
    downstream_df = pd.DataFrame(downstream_data)
    
    return upstream_df, downstream_df

def main():
    """Main Streamlit application."""
    st.title("🔍 dbt Impact Analysis - Cytoscape Version")
    st.markdown("**Interactive Cytoscape.js Graphs** | Analyze dbt dependencies with professional interactive visualizations")
    
    if not CYTOSCAPE_AVAILABLE:
        st.error("st-link-analysis is not available. Please install it to use this version.")
        st.code("pip install st-link-analysis")
        st.info("This component provides interactive graph visualization using Cytoscape.js")
        return
    
    # Load graph data
    with st.spinner("Loading knowledge graph..."):
        graph = load_graph_data()
        metadata = load_metadata()
    
    if not graph:
        st.error("Failed to load graph data. Please check your data files.")
        return
    
    # Sidebar
    st.sidebar.title("Impact Analysis Configuration")
    
    # Display metadata
    if metadata:
        st.sidebar.subheader("📊 Graph Statistics")
        stats = metadata.get('statistics', {})
        st.sidebar.metric("Total Nodes", stats.get('total_nodes', 0))
        st.sidebar.metric("Total Edges", stats.get('total_edges', 0))
        st.sidebar.metric("Node Types", len(stats.get('node_types', {})))
    
    # Node type selection
    node_types = ['model', 'source', 'test', 'exposure', 'metric', 'macro', 'seed', 'snapshot', 'analysis']
    node_type = st.sidebar.selectbox("Select Node Type", node_types)
    
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
    
    # Add aggregated view option
    aggregate_view = st.sidebar.checkbox(
        "Aggregated View", 
        value=True if node_type == 'source' else False,
        help="Group similar nodes by source and package"
    )
    
    # Cytoscape-specific options
    st.sidebar.subheader("🎨 Cytoscape Layout Options")
    
    layout_algorithm = st.sidebar.selectbox(
        "Layout Algorithm",
        list(LAYOUT_ALGORITHMS.keys()),
        format_func=lambda x: f"{x} - {LAYOUT_ALGORITHMS[x]}",
        index=0,  # Default to "dagre"
        help="Choose the graph layout algorithm"
    )
    
    # Perform impact analysis
    with st.spinner("Analyzing impact..."):
        result = create_impact_subgraph(
            graph, selected_node, upstream_depth, downstream_depth,
            include_tests, include_columns, include_packages
        )
        
        if result is None:
            return
            
        subgraph, upstream_nodes, downstream_nodes = result
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
                for pkg in sorted(metrics['upstream_packages']):
                    st.write(f"• {pkg}")
        
        with col2:
            if metrics['downstream_packages']:
                st.write("**Downstream Packages:**")
                for pkg in sorted(metrics['downstream_packages']):
                    st.write(f"• {pkg}")
    
    # Main analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Impact Analysis", "🎨 Cytoscape Visualization", "📋 Node Details"])
    
    with tab1:
        st.subheader("📈 Impact Analysis Tables")
        
        # Get impact tables
        upstream_df, downstream_df = create_impact_tables(
            graph, upstream_nodes, downstream_nodes, node_type, aggregate_view
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"⬆️ Upstream Dependencies ({len(upstream_df)})")
            if len(upstream_df) > 0:
                st.dataframe(upstream_df, use_container_width=True)
                
                # Package breakdown chart
                if 'dbt_project' in upstream_df.columns:
                    package_counts = upstream_df['dbt_project'].value_counts()
                    fig = px.pie(values=package_counts.values, names=package_counts.index,
                               title="Upstream Packages")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No upstream dependencies found.")
        
        with col2:
            st.subheader(f"⬇️ Downstream Impact ({len(downstream_df)})")
            if len(downstream_df) > 0:
                st.dataframe(downstream_df, use_container_width=True)
                
                # Package breakdown chart
                if 'dbt_project' in downstream_df.columns:
                    package_counts = downstream_df['dbt_project'].value_counts()
                    fig = px.pie(values=package_counts.values, names=package_counts.index,
                               title="Downstream Packages")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No downstream impact found.")
    
    with tab2:
        st.subheader("🎨 Interactive Cytoscape Visualization")
        
        # Create and display the Cytoscape graph
        with st.spinner(f"Creating {LAYOUT_ALGORITHMS[layout_algorithm]} visualization..."):
            try:
                config = create_cytoscape_visualization(
                    graph, subgraph, selected_node, upstream_nodes, downstream_nodes, layout_algorithm
                )
                
                if config:
                    # Create custom node and edge styles for better visualization
                    from st_link_analysis import NodeStyle, EdgeStyle
                    
                    # Define node styles for different impact types based on node type labels
                    node_styles = [
                        NodeStyle(
                            label="source",
                            color="#f5a142",
                            caption="name",
                            icon="source"
                        ),
                        NodeStyle(
                            label="model", 
                            color="#4287f5",
                            caption="name",
                            icon="schema"
                        ),
                        NodeStyle(
                            label="test",
                            color="#42f59e",
                            caption="name",
                            icon="verified"
                        ),
                        NodeStyle(
                            label="seed",
                            color="#f542a1",
                            caption="name",
                            icon="data_table"
                        ),
                        NodeStyle(
                            label="snapshot",
                            color="#a142f5",
                            caption="name",
                            icon="camera"
                        )
                    ]
                    
                    # Define edge styles
                    edge_styles = [
                        EdgeStyle(
                            label="depends_on",
                            color="#666666",
                            directed=True,
                            curve_style="bezier"
                        )
                    ]
                    
                    # Display the interactive graph
                    result = st_link_analysis(
                        elements=config["elements"],
                        layout=config["layout"],
                        node_styles=node_styles,
                        edge_styles=edge_styles,
                        key="dbt_impact_graph",
                        height=600
                    )
                    
                    # Show interaction results
                    if result:
                        st.subheader("🔍 Graph Interactions")
                        with st.expander("View interaction details", expanded=False):
                            st.json(result)
                    
                    # Visualization info
                    with st.expander("🎨 Visualization Details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**🔴 Selected Node**")
                            st.markdown("The node you're analyzing")
                            
                        with col2:
                            st.markdown("**🟠 Upstream Dependencies**")
                            st.markdown("Nodes this depends on")
                            
                        with col3:
                            st.markdown("**🔵 Downstream Impact**")
                            st.markdown("Nodes that depend on this")
                        
                        st.markdown("---")
                        st.markdown("**Layout Algorithms:**")
                        for algo, desc in LAYOUT_ALGORITHMS.items():
                            current = "👉 " if algo == layout_algorithm else "   "
                            st.markdown(f"{current}**{algo}**: {desc}")
                        
                        st.markdown("---")
                        st.markdown("**Interactive Features:**")
                        st.markdown("- 🖱️ **Drag** nodes to reposition")
                        st.markdown("- 🔍 **Scroll** to zoom in/out") 
                        st.markdown("- 📱 **Click** nodes/edges for details")
                        st.markdown("- 🎯 **Double-click** for node actions")
                        st.markdown("- 🖼️ **Right-click** for context menu")
                        
                        st.markdown("---")
                        st.markdown("**Node Shapes:**")
                        st.markdown("- 🔵 **Ellipse**: Models")
                        st.markdown("- 🟩 **Rectangle**: Tests")
                        st.markdown("- 💎 **Diamond**: Sources (consolidated)")
                        st.markdown("- 🔺 **Triangle**: Columns")
                        st.markdown("- ⭐ **Star**: Exposures")
                        st.markdown("- ⬢ **Hexagon**: Metrics")
                        
                else:
                    st.error("Failed to generate Cytoscape visualization")
                    
            except Exception as e:
                st.error(f"Error creating visualization: {e}")
                st.info("Try adjusting the layout algorithm or reducing the analysis depth.")
    
    with tab3:
        st.subheader(f"Selected {node_type.title()} Details")
        
        # Get node attributes with safety check
        if selected_node not in graph.nodes:
            st.error(f"Selected node '{selected_node}' not found in graph. Please regenerate the graph data.")
            st.info("Run: `python manifest_parser.py path/to/manifest.json --storage-dir data`")
            return
            
        node_attrs = graph.nodes[selected_node]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information**")
            st.write(f"**ID:** {selected_node}")
            st.write(f"**Name:** {node_attrs.get('name', 'N/A')}")
            st.write(f"**Type:** {node_attrs.get('node_type', 'N/A')}")
            st.write(f"**Package:** {node_attrs.get('package_name', 'N/A')}")
        
        with col2:
            st.write("**Technical Details**")
            if 'database' in node_attrs:
                st.write(f"**Database:** {node_attrs['database']}")
            if 'schema' in node_attrs:
                st.write(f"**Schema:** {node_attrs['schema']}")
            if 'materialization' in node_attrs:
                st.write(f"**Materialization:** {node_attrs['materialization']}")
        
        if 'description' in node_attrs and node_attrs['description']:
            st.write("**Description**")
            st.write(node_attrs['description'])
        
        if 'tags' in node_attrs and node_attrs['tags']:
            st.write("**Tags**")
            st.write(", ".join(node_attrs['tags']))

if __name__ == "__main__":
    main()
