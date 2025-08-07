# Consolidated dbt Knowledge Graph Builder

The `manifest_parser.py` script has been enhanced to include all storage and export functionality previously split between `manifest_parser.py` and `graph_storage.py`. This provides a streamlined workflow for building knowledge graphs from dbt manifests.

## Features

- **Parse dbt manifest.json** and extract nodes and relationships
- **Automatic storage** in multiple formats (JSON, pickle, metadata)
- **Visualization export** (GraphML, GML, web-ready JSON)
- **Fast loading** from stored graphs
- **Source name fix** included (formats source names as `source_name.table_name`)

## Output Files

### Primary Storage (stored in `--storage-dir`, default: `data/`)
- `nodes.json` - All nodes with properties
- `edges.json` - All relationships between nodes  
- `knowledge_graph.gpickle` - NetworkX graph (fastest loading)
- `graph_metadata.json` - Creation time, statistics, file info

### Visualization Files (when using `--export-viz`)
- `visualization/knowledge_graph.graphml` - For Gephi, Cytoscape
- `visualization/knowledge_graph.gml` - For various graph tools
- `visualization/visualization_data.json` - For web visualization

## Usage Examples

### Basic Workflow (recommended)
```bash
# Parse manifest and store all outputs
python manifest_parser.py code_docs/manifest.json

# Parse and export visualization files too
python manifest_parser.py code_docs/manifest.json --export-viz

# Use custom storage directory
python manifest_parser.py code_docs/manifest.json --storage-dir my_output
```

### Advanced Usage
```bash
# Load existing graph (fast)
python manifest_parser.py code_docs/manifest.json --load-only

# Export to specific files (legacy compatibility)
python manifest_parser.py code_docs/manifest.json --legacy-export nodes.json,edges.json

# Verbose logging
python manifest_parser.py code_docs/manifest.json --verbose
```

### Programmatic Usage
```python
from manifest_parser import ManifestParser

# Simple workflow - parse and store everything
parser = ManifestParser('path/to/manifest.json', 'output_dir')
graph = parser.build_and_store_graph()

# Load existing graph
parser = ManifestParser('path/to/manifest.json', 'output_dir') 
graph = parser.load_graph()

# Export visualizations
parser.export_for_visualization('viz_output')
```

## Migration from graph_storage.py

If you were using `graph_storage.py`, simply replace:

**Old:**
```bash
python graph_storage.py manifest.json --export-viz
```

**New:**
```bash
python manifest_parser.py manifest.json --export-viz
```

The consolidated script provides all the same functionality with improved performance and consistency.

## Running Impact Analysis

After generating the knowledge graph, run the Streamlit impact analysis app:

```bash
# Generate knowledge graph first
python manifest_parser.py code_docs/manifest.json --storage-dir data --export-viz

# Run the impact analysis app
streamlit run impact_analysis_app.py
```

The app will be available at http://localhost:8501 and provides:
- Interactive dependency visualization
- Risk assessment for changes
- Package impact analysis
- Detailed node information

## Source Name Format

Source nodes now use the correct naming format:
- **Before**: `account_edp_optimized`
- **After**: `account_optimized.account_edp_optimized`

This applies to all source nodes in the format `source_name.table_name`.
