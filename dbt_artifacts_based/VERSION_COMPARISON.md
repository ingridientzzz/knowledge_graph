# Impact Analysis App Version Comparison

## 📊 Three Versions Available

### `impact_analysis_app.py` - Standard Version
**Best for**: Local development, environments with custom Streamlit components

**Visualization**: streamlit-agraph (React-based network visualization)
- Interactive physics-based layout
- Vis.js powered graph rendering
- Custom node shapes and styling
- Advanced interaction capabilities

**Requirements**:
```txt
streamlit>=1.28.0
streamlit-agraph>=0.0.39
plotly>=5.15.0
pandas>=1.5.0
networkx>=2.8.0
```

### `impact_analysis_app_plotly.py` - Snowflake Compatible Version
**Best for**: Snowflake Streamlit deployments, restricted environments

**Visualization**: Plotly Graph Objects (Native Plotly network visualization)
- Spring layout algorithm from NetworkX
- Pure Plotly-based interactive graphs
- Full zoom, pan, hover capabilities
- No external component dependencies

**Requirements**:
```txt
streamlit>=1.28.0
plotly>=5.15.0
pandas>=1.5.0
networkx>=2.8.0
```

### `impact_analysis_app_graphviz.py` - Professional Layouts Version
**Best for**: High-quality static visualizations, publication-ready graphs

**Visualization**: PyGraphviz (Professional graph layout algorithms)
- Multiple layout algorithms (dot, neato, fdp, sfdp, circo, twopi)
- Publication-quality static graph rendering
- Professional node shapes and styling
- Downloadable PNG outputs

**Requirements**:
```txt
streamlit>=1.28.0
plotly>=5.15.0
pandas>=1.5.0
networkx>=2.8.0
pygraphviz>=1.9
```

## 🔄 Feature Comparison

| Feature | Standard Version | Plotly Version | GraphViz Version | Notes |
|---------|------------------|----------------|------------------|-------|
| **Node Visualization** | ✅ Custom shapes | ✅ Plotly symbols | ✅ Professional shapes | All support type-based styling |
| **Interactive Pan/Zoom** | ✅ Full support | ✅ Full support | ❌ Static image | GraphViz creates static PNG |
| **Hover Tooltips** | ✅ HTML tooltips | ✅ Text tooltips | ❌ No tooltips | GraphViz is static |
| **Node Consolidation** | ✅ Source grouping | ✅ Source grouping | ✅ Source grouping | Identical logic |
| **Impact Analysis** | ✅ Complete | ✅ Complete | ✅ Complete | Identical analysis features |
| **Risk Assessment** | ✅ Complete | ✅ Complete | ✅ Complete | Same risk calculation |
| **Package Charts** | ✅ Plotly pie charts | ✅ Plotly pie charts | ✅ Plotly pie charts | Identical |
| **Data Tables** | ✅ Streamlit tables | ✅ Streamlit tables | ✅ Streamlit tables | Identical |
| **Critical Nodes** | ✅ Treemap viz | ✅ Treemap viz | ✅ Treemap viz | Identical |
| **Layout Algorithms** | ⚡ Physics only | ⚡ Spring only | ✅ 6 algorithms | GraphViz has most options |
| **Graph Quality** | ⚡ Good | ⚡ Good | ✅ Excellent | GraphViz professional quality |
| **Download/Export** | ❌ No export | ❌ No export | ✅ PNG download | GraphViz exports images |
| **Performance** | ⚡ Good | ✅ Better | ✅ Fast rendering | All perform well |

## 🎯 Choosing the Right Version

### Use Standard Version When:
- ✅ Developing locally
- ✅ Need advanced physics simulations
- ✅ Want maximum customization
- ✅ Environment supports custom components

### Use Plotly Version When:
- ✅ Deploying to Snowflake Streamlit
- ✅ Environment restricts custom components
- ✅ Need better performance on large graphs
- ✅ Want fewer dependencies

### Use GraphViz Version When:
- ✅ Need publication-quality graphs
- ✅ Want professional layout algorithms
- ✅ Need to export/download visualizations
- ✅ Creating documentation or presentations
- ✅ Prefer static over interactive graphs

## 🚀 Quick Start Guide

### For Local Development:
```bash
# Install standard requirements
pip install -r requirements.txt
streamlit run impact_analysis_app.py
```

### For Snowflake Streamlit:
```bash
# Install plotly requirements
pip install -r requirements_plotly.txt
streamlit run impact_analysis_app_plotly.py
```

### For Professional Layouts:
```bash
# Install graphviz and pygraphviz
brew install graphviz  # macOS
# sudo apt-get install graphviz graphviz-dev  # Ubuntu
pip install -r requirements_graphviz.txt
streamlit run impact_analysis_app_graphviz.py
```

## 📋 Migration Notes

### Code Structure
All versions share:
- ✅ Identical data loading functions
- ✅ Same impact analysis logic
- ✅ Same table generation
- ✅ Same risk assessment
- ✅ Same user interface layout

### Key Differences
Only the visualization rendering differs:
- Standard: `streamlit_agraph.agraph(nodes, edges, config)`
- Plotly: `st.plotly_chart(plotly_figure)`
- GraphViz: `st.image(base64_encoded_png)`

### Data Compatibility
- ✅ All use same data files
- ✅ All support same analysis features
- ✅ Same configuration options
- ✅ Identical analysis quality

## 🎨 Visual Differences

### Layout Algorithm
- **Standard**: Vis.js force-directed physics (more dynamic)
- **Plotly**: NetworkX spring layout (more stable)
- **GraphViz**: 6 professional algorithms (dot, neato, fdp, sfdp, circo, twopi)

### Node Appearance
- **Standard**: Custom shapes (diamond, triangle, etc.)
- **Plotly**: Plotly symbols (similar but standardized)
- **GraphViz**: Professional shapes (diamond, ellipse, box, etc.)

### Interactivity
- **Standard**: Physics-based node dragging
- **Plotly**: Static layout with zoom/pan
- **GraphViz**: Static image (no interaction)

### Performance
- **Standard**: Good for <100 nodes
- **Plotly**: Better for 100+ nodes
- **GraphViz**: Excellent for any size (static rendering)

### Output Quality
- **Standard**: Good interactive quality
- **Plotly**: Good interactive quality
- **GraphViz**: Excellent publication quality

All three versions provide comprehensive analysis capabilities with different visualization strengths!
