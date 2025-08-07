# dbt Impact Analysis - igraph Setup Guide

This guide covers the setup and usage of the **igraph version** of the dbt Impact Analysis application.

## 🎯 Features

- **Professional Graph Layouts**: Uses python-igraph's advanced layout algorithms
- **Multiple Layout Algorithms**: Choose from Fruchterman-Reingold, Kamada-Kawai, DrL, and more
- **High-Quality Visualization**: Optimized node positioning and edge routing
- **Interactive Plotly Charts**: Zoom, pan, and hover interactions
- **Performance**: Efficient layout computation for large graphs

## 📦 Dependencies

The igraph version requires additional dependencies for graph layout computation:

```bash
pip install python-igraph
```

On some systems, you may need to install system-level dependencies:

### macOS (with Homebrew)
```bash
brew install igraph
```

### Ubuntu/Debian
```bash
sudo apt-get install libigraph0-dev
```

### Windows
```bash
# python-igraph should install without additional dependencies
pip install python-igraph
```

## 🚀 Installation

1. **Clone or navigate to the project directory**
2. **Create virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements_igraph.txt
   ```

4. **Ensure you have processed data**:
   - Run `manifest_parser.py` to generate the knowledge graph
   - Check that `data/` directory contains the graph files

## 🖥️ Running the App

```bash
streamlit run impact_analysis_app_igraph.py
```

The app will be available at `http://localhost:8501`

## 🎨 Layout Algorithms

The igraph version supports multiple layout algorithms:

| Algorithm | Code | Best For | Performance |
|-----------|------|----------|-------------|
| **Fruchterman-Reingold** | `fr` | General purpose, good separation | Medium |
| **Kamada-Kawai** | `kk` | High-quality layouts, symmetric | Slower |
| **DrL** | `drl` | Large graphs, fast computation | Fast |
| **Large Graph Layout** | `lgl` | Very large graphs (1000+ nodes) | Very Fast |
| **Circular** | `circle` | Simple circular arrangement | Very Fast |
| **Reingold-Tilford** | `rt` | Tree-like structures | Fast |

## 🔧 Troubleshooting

### Installation Issues

**Error: Failed building wheel for python-igraph**
- Solution: Install system dependencies (see Dependencies section above)

**Error: Could not find igraph library**
- macOS: `brew install igraph`
- Ubuntu: `sudo apt-get install libigraph0-dev`

### Runtime Issues

**Error: Layout algorithm not found**
- Check that you're using supported algorithm codes: `fr`, `kk`, `drl`, `lgl`, `circle`, `rt`

**Poor layout quality**
- Try different algorithms: `kk` for high quality, `drl` for speed
- Larger graphs may benefit from `lgl` or `drl`

## 📊 Performance Tips

1. **For small graphs (< 50 nodes)**: Use `kk` or `fr` for best quality
2. **For medium graphs (50-200 nodes)**: Use `fr` or `drl`
3. **For large graphs (200+ nodes)**: Use `drl` or `lgl`
4. **For very large graphs (1000+ nodes)**: Use `lgl` only

## 🆚 Comparison with Other Versions

| Feature | igraph | streamlit-agraph | Plotly | Cytoscape |
|---------|--------|------------------|--------|-----------|
| Layout Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Performance | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Interactivity | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Setup Complexity | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 📁 File Structure

```
dbt_artifacts_based/
├── impact_analysis_app_igraph.py     # Main igraph application
├── requirements_igraph.txt           # Dependencies for igraph version
├── IGRAPH_SETUP.md                  # This setup guide
└── data/                            # Generated graph data
    ├── knowledge_graph.gpickle
    ├── nodes.json
    └── edges.json
```

## 🔗 Related Files

- `impact_analysis_app.py` - Original streamlit-agraph version
- `impact_analysis_app_plotly.py` - Plotly-only version
- `impact_analysis_app_cytoscape.py` - Cytoscape.js version
- `VERSION_COMPARISON.md` - Detailed comparison of all versions
