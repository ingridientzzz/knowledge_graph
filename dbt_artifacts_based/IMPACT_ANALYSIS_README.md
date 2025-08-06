# 🔍 dbt Impact Analysis - Streamlit App

A comprehensive Streamlit application for visualizing and analyzing the impact of changes in dbt projects using the knowledge graph built from `manifest.json`.

## 🌟 Features

### 📊 **Graph Overview**
- **12,104 nodes** across 10 different resource types
- **16,397 relationships** showing dependencies and connections
- **Real-time statistics** from your dbt project

### 🎯 **Impact Analysis**
- **Upstream Dependencies**: See what your selected resource depends on
- **Downstream Impact**: Understand what will be affected by changes
- **Risk Assessment**: Automatic risk level calculation (Low/Medium/High)
- **Package Impact**: Track changes across dbt packages
- **Depth Control**: Configure how many levels of dependencies to analyze

### 🎨 **Interactive Visualization**
- **Node-Link Diagrams**: Interactive graph visualization with agraph
- **Color-Coded Nodes**: Different colors for each resource type
- **Shape Differentiation**: Unique shapes for models, sources, tests, etc.
- **Impact Highlighting**: Selected nodes and their dependencies are highlighted
- **Zoom and Pan**: Explore large dependency graphs interactively

### 📋 **Detailed Analysis**
- **Searchable Tables**: Filter and search through impacted resources
- **Type Breakdowns**: Pie charts showing distribution of affected resource types
- **Critical Nodes**: Identify highly connected downstream resources
- **Package Distribution**: See which packages are most affected

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have the parsed graph data
python graph_storage.py path/to/manifest.json --storage-dir data
```

### Launch the App
```bash
# Method 1: Using the launcher script
python run_app.py

# Method 2: Direct Streamlit command
streamlit run impact_analysis_app.py --server.port 8502

# Method 3: Using the virtual environment
source /path/to/.venv/bin/activate
streamlit run impact_analysis_app.py
```

### Demo Analysis
```bash
# Run the demo to see key insights
python demo_analysis.py
```

## 📈 Key Insights from Demo

Based on the sample analysis of your dbt project:

### 🔥 **High-Impact Models**
1. **accelerator_consulting_survey_response_transform**
   - 262 downstream dependencies
   - Affects 254 columns, 4 models, 4 tests
   - **High Risk** for changes

2. **cpp_accelerator_survey_response_transform**
   - 155 downstream dependencies
   - **High Risk** impact level

### 📡 **Critical Sources**
1. **cpp_step_survey_session**
   - 79 downstream dependencies
   - High impact source requiring careful change management

### 📦 **Package Distribution**
- **Elementary**: 715 resources (monitoring/testing)
- **dbt**: 382 resources (core package)
- **customer_segments**: 270 resources
- **rhs**: 207 resources
- **tflex**: 184 resources

## 🎛️ App Interface Guide

### Sidebar Controls

#### 🎯 **Node Selection**
- **Node Type**: Choose from models, sources, tests, exposures, etc.
- **Search**: Filter nodes by name or ID
- **Selection**: Pick the specific resource to analyze

#### 🔧 **Analysis Options**
- **Upstream Depth** (1-10): How many levels of dependencies to trace backwards
- **Downstream Depth** (1-10): How many levels of impact to trace forwards
- **Include Tests**: Show/hide test relationships
- **Include Columns**: Show/hide column-level dependencies
- **Include Packages**: Show/hide package nodes

### Main Interface

#### 📊 **Metrics Dashboard**
- **Upstream Dependencies**: Total number of upstream resources
- **Downstream Impact**: Total number of affected downstream resources
- **Total Affected Nodes**: Complete impact scope
- **Risk Level**: Automated risk assessment with color coding

#### ⚠️ **Risk Assessment**
- **Risk Factors**: Specific warnings about the change impact
- **Package Impact**: Which packages will be affected
- **Critical Nodes**: Highly connected downstream resources

#### 🎨 **Visualization**
- **Interactive Graph**: Explore dependencies visually
- **Legend**: Understand node colors and shapes
- **Zoom/Pan**: Navigate large dependency networks

#### 📋 **Detailed Analysis**
- **Upstream Tab**: Table of all upstream dependencies
- **Downstream Tab**: Table of all downstream impacts
- **Node Details Tab**: Comprehensive information about the selected resource

## 🎨 Visual Legend

### Node Colors
- 🔴 **Selected Node**: The resource you're analyzing
- 🟠 **Upstream**: Dependencies (what the selected node depends on)
- 🔵 **Downstream**: Impact (what depends on the selected node)
- 🟦 **Models**: Blue for dbt models
- 🟢 **Tests**: Green for dbt tests
- 🟡 **Sources**: Orange for data sources
- 🟣 **Columns**: Pink for individual columns

### Node Shapes
- ● **Models**: Circular nodes
- ▲ **Tests**: Triangle nodes
- ♦ **Sources**: Diamond nodes
- ■ **Columns**: Square nodes
- ⭐ **Exposures**: Star nodes
- ⬢ **Metrics**: Hexagon nodes

## 📊 Risk Assessment Levels

### 🟢 **Low Risk**
- ≤5 downstream dependencies
- Limited package impact
- Standard deployment process recommended

### 🟡 **Medium Risk**
- 6-10 downstream dependencies
- Multi-package impact
- Enhanced testing and stakeholder notification recommended

### 🔴 **High Risk**
- >10 downstream dependencies
- Affects critical/highly connected nodes
- Comprehensive testing, coordination, and rollback planning required

## 🛠️ Advanced Features

### Package Impact Analysis
- See which dbt packages are affected by your changes
- Understand cross-package dependencies
- Plan coordinated deployments

### Column-Level Impact
- Trace column lineage through transformations
- Understand data flow at the granular level
- Identify breaking schema changes

### Critical Node Detection
- Automatically identify highly connected resources
- Focus on nodes with >10 downstream dependencies
- Prioritize testing for critical infrastructure

## 📁 File Structure

```
dbt_artifacts_based/
├── impact_analysis_app.py     # Main Streamlit application
├── run_app.py                 # Simple launcher script
├── demo_analysis.py           # Demo showcasing key features
├── data/                      # Generated graph data
│   ├── knowledge_graph.gpickle
│   ├── nodes.json
│   ├── edges.json
│   └── graph_metadata.json
└── visualization/             # Export formats
    ├── knowledge_graph.graphml
    ├── knowledge_graph.gml
    └── visualization_data.json
```

## 🔧 Configuration

### Streamlit Configuration
The app uses these default settings:
- **Port**: 8502 (to avoid conflicts)
- **Layout**: Wide mode for better visualization
- **Caching**: Enabled for graph data loading

### Performance Optimization
- **Graph Caching**: Graph data is cached using `@st.cache_data`
- **Selective Loading**: Only load required data for analysis
- **Depth Limiting**: Control analysis depth to manage performance

## 🐛 Troubleshooting

### Common Issues

1. **No graph data found**
   ```bash
   # Solution: Run the manifest parser first
   python graph_storage.py path/to/manifest.json --storage-dir data
   ```

2. **Port already in use**
   ```bash
   # Solution: Use a different port
   streamlit run impact_analysis_app.py --server.port 8503
   ```

3. **Memory issues with large graphs**
   - Reduce upstream/downstream depth
   - Disable column inclusion for initial analysis
   - Use the demo script for quick insights

4. **Missing dependencies**
   ```bash
   # Solution: Install required packages
   pip install streamlit streamlit-agraph plotly pandas networkx
   ```

## 🔮 Future Enhancements

- **Column Lineage**: Enhanced column-level dependency tracking
- **Time-based Analysis**: Historical impact trends
- **Custom Risk Rules**: Configurable risk assessment criteria
- **Export Capabilities**: Save analysis results and visualizations
- **Collaboration Features**: Share impact analyses with team members

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify that graph data has been generated from manifest.json
4. Review the demo script output for baseline functionality

## 🎉 Success Metrics

The app successfully handles:
- **Large Scale**: 12,000+ nodes, 16,000+ edges
- **Multiple Resource Types**: 10 different dbt resource types
- **Real-time Analysis**: Interactive dependency exploration
- **Risk Assessment**: Automated change impact evaluation
- **Visual Exploration**: Interactive graph visualization

Perfect for teams managing complex dbt projects with extensive dependencies and multiple packages!