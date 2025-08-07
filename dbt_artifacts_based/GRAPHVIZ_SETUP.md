# GraphViz Setup Guide

## Installing PyGraphviz for Professional Graph Layouts

The GraphViz version (`impact_analysis_app_graphviz.py`) requires PyGraphviz, which depends on the system GraphViz library.

### 🍎 macOS Installation

```bash
# Install GraphViz using Homebrew
brew install graphviz

# Install Python package
pip install pygraphviz

# Or install all requirements
pip install -r requirements_graphviz.txt
```

### 🐧 Ubuntu/Debian Installation

```bash
# Install GraphViz and development headers
sudo apt-get update
sudo apt-get install graphviz graphviz-dev

# Install Python package
pip install pygraphviz

# Or install all requirements
pip install -r requirements_graphviz.txt
```

### 🎯 CentOS/RHEL Installation

```bash
# Install GraphViz and development tools
sudo yum install graphviz graphviz-devel

# Install Python package
pip install pygraphviz

# Or install all requirements
pip install -r requirements_graphviz.txt
```

### 🪟 Windows Installation

#### Option 1: Using Conda (Recommended)
```bash
# Install from conda-forge (includes GraphViz)
conda install -c conda-forge pygraphviz

# Install other requirements
pip install streamlit plotly pandas networkx
```

#### Option 2: Manual Installation
1. Download GraphViz from: https://graphviz.org/download/
2. Install GraphViz and add to PATH
3. Install PyGraphviz:
```bash
pip install pygraphviz
```

### 🐳 Docker Installation

```dockerfile
FROM python:3.9-slim

# Install GraphViz
RUN apt-get update && \
    apt-get install -y graphviz graphviz-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_graphviz.txt .
RUN pip install -r requirements_graphviz.txt

# Copy your app files
COPY . /app
WORKDIR /app

EXPOSE 8501
CMD ["streamlit", "run", "impact_analysis_app_graphviz.py"]
```

### ✅ Verify Installation

Test your installation:

```python
import pygraphviz as pgv
print("PyGraphviz version:", pgv.__version__)

# Test basic functionality
G = pgv.AGraph()
G.add_edge('A', 'B')
G.layout()
print("✅ PyGraphviz is working correctly!")
```

### 🚨 Troubleshooting

#### Issue: "GraphViz not found"
- **Solution**: Ensure GraphViz is installed and in your PATH
- **macOS**: `brew list graphviz` should show installed
- **Linux**: `which dot` should return a path

#### Issue: "Failed building wheel for pygraphviz"
- **Solution**: Install development headers
- **Ubuntu**: `sudo apt-get install graphviz-dev`
- **CentOS**: `sudo yum install graphviz-devel`

#### Issue: "Cannot import name 'AGraph'"
- **Solution**: Reinstall PyGraphviz after installing GraphViz
```bash
pip uninstall pygraphviz
pip install pygraphviz
```

### 🎨 Layout Algorithms Available

Once installed, you'll have access to these professional layout algorithms:

1. **dot** - Hierarchical (top-to-bottom) - Perfect for dependency trees
2. **neato** - Spring Model - Good for general graphs
3. **fdp** - Force-Directed - Better spacing for dense graphs  
4. **sfdp** - Scalable Force-Directed - Best for large graphs (>1000 nodes)
5. **circo** - Circular Layout - Beautiful for small, connected graphs
6. **twopi** - Radial Layout - Great for star-like patterns

### 🚀 Ready to Use!

Once PyGraphviz is installed, run:

```bash
streamlit run impact_analysis_app_graphviz.py
```

Enjoy professional-quality graph visualizations! 🎯
