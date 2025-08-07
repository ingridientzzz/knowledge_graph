# Cytoscape Setup Guide

## Installing st-link-analysis for Interactive Graph Visualization

The Cytoscape version (`impact_analysis_app_cytoscape.py`) uses `st-link-analysis`, a Streamlit component based on Cytoscape.js for professional interactive graph visualization.

### 🚀 Quick Installation

```bash
# Install st-link-analysis
pip install st-link-analysis

# Or install all requirements
pip install -r requirements_cytoscape.txt
```

### 📦 What is st-link-analysis?

`st-link-analysis` is a modern Streamlit component that provides:
- **Interactive graph visualization** using Cytoscape.js
- **Multiple layout algorithms** (dagre, fcose, cola, etc.)
- **Professional node/edge manipulation**
- **Snowflake Streamlit compatibility**

### ✅ Verify Installation

Test your installation:

```python
import streamlit as st
from st_link_analysis import st_link_analysis

# Create a simple test graph
elements = [
    {"data": {"id": "A", "label": "Node A"}},
    {"data": {"id": "B", "label": "Node B"}},
    {"data": {"id": "edge1", "source": "A", "target": "B"}}
]

layout = {"name": "dagre"}

# Display the graph
result = st_link_analysis(elements=elements, layout=layout, key="test")
print("✅ st-link-analysis is working correctly!")
```

### 🎨 Layout Algorithms Available

Once installed, you'll have access to these professional layout algorithms:

1. **dagre** - Hierarchical (DAG-based) - Perfect for dependency flows
2. **breadthfirst** - Breadth-first tree layout
3. **circle** - Circular arrangement
4. **concentric** - Concentric circles
5. **grid** - Grid-based positioning
6. **random** - Random positions
7. **cose** - Compound Spring Embedder
8. **fcose** - Fast Compound Spring Embedder (best for large graphs)
9. **cola** - Constraint-based layout

### 🏔️ Snowflake Streamlit Compatibility

**Great News!** As of **August 2025**, Snowflake Streamlit supports custom components that don't require external services:

> *"Custom components are now supported in Streamlit in Snowflake. Currently, Streamlit in Snowflake only supports custom components that don't require making calls to external services."*

**st-link-analysis is compatible** because:
- ✅ Pure frontend JavaScript component
- ✅ No external API calls required
- ✅ Self-contained Cytoscape.js implementation
- ✅ Works entirely within the Streamlit environment

### 🎯 Interactive Features

The Cytoscape version provides rich interactivity:

- **🖱️ Node Dragging** - Reposition nodes by dragging
- **🔍 Zoom & Pan** - Scroll to zoom, drag to pan
- **📱 Click Events** - Click nodes/edges for details
- **🎯 Double-click Actions** - Node expansion/removal
- **🖼️ Context Menus** - Right-click for options
- **📏 Dynamic Layouts** - Switch layouts on the fly

### 🔧 Configuration Options

The app provides several customization options:

#### Layout Selection
Choose from 9 different algorithms optimized for different graph types:
- **dagre** - Best for hierarchical dbt dependency graphs
- **fcose** - Optimal for large, complex graphs
- **circle** - Great for smaller, connected components

#### Visual Styling
- **Node shapes** based on dbt resource types (diamond=sources, rectangle=tests, etc.)
- **Color coding** for impact analysis (red=selected, orange=upstream, blue=downstream)
- **Dynamic sizing** based on importance and impact level

### 🚨 Troubleshooting

#### Issue: "st_link_analysis not found"
```bash
pip install st-link-analysis
```

#### Issue: Layout not displaying correctly
- Try switching to a different layout algorithm
- Check that your graph has valid node and edge data
- Ensure nodes have unique IDs

#### Issue: Performance with large graphs
- Use **fcose** or **cose** layouts for better performance
- Consider filtering to reduce node count
- Enable aggregated view for sources

### 🎊 Why Choose the Cytoscape Version?

The Cytoscape version is **the best choice for Snowflake Streamlit** because it provides:

1. **✅ Full Interactivity** - Similar to streamlit-agraph
2. **✅ Snowflake Compatible** - Works in restricted environments  
3. **✅ Professional Quality** - Cytoscape.js is industry standard
4. **✅ Rich Layouts** - 9 algorithms vs 1 in other versions
5. **✅ Active Development** - Regular updates and community support

### 🚀 Ready to Use!

Once st-link-analysis is installed, run:

```bash
streamlit run impact_analysis_app_cytoscape.py
```

Enjoy the best of both worlds: **interactive visualization + Snowflake compatibility**! 🎯

### 📚 Additional Resources

- **st-link-analysis GitHub**: https://github.com/AlrasheedA/st-link-analysis
- **Cytoscape.js Documentation**: https://js.cytoscape.org/
- **Streamlit Components**: https://docs.streamlit.io/develop/concepts/custom-components
