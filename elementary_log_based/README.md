# dbt Elementary Knowledge Graph

A comprehensive knowledge graph solution for analyzing dbt projects using Elementary observability data from Snowflake.

## Overview

This project creates a knowledge graph from dbt Elementary tables, enabling powerful analysis of:
- **Data Lineage**: Trace data flow from sources to final models
- **Test Coverage**: Analyze data quality test coverage across your project
- **Performance Monitoring**: Identify slow-running models and optimization opportunities
- **Impact Analysis**: Understand the downstream effects of changes
- **Dependency Analysis**: Visualize complex model relationships

## Features

### 🏗️ Graph Construction
- **Entity Extraction**: Automatically extracts models, tests, sources, columns, and invocations
- **Relationship Mapping**: Builds comprehensive dependency graphs
- **Metadata Enrichment**: Adds rich metadata as node and edge properties

### 📊 Analytics & Insights
- **Lineage Tracing**: Full upstream/downstream dependency analysis
- **Test Coverage Analysis**: Identify models lacking adequate testing
- **Performance Profiling**: Find slow models and execution bottlenecks
- **Data Quality Monitoring**: Track test failures and data issues
- **Complexity Analysis**: Measure dependency complexity and circular dependencies

### 💾 Flexible Storage
- **File System**: Store graphs as NetworkX pickle files
- **Neo4j Integration**: Full graph database support for complex queries
- **Export Options**: Export to GEXF, GraphML, JSON, and CSV formats

### 🔍 Advanced Querying
- **Pattern Matching**: Find models by tags, schemas, or patterns
- **Path Analysis**: Discover longest dependency chains
- **Schema Dependencies**: Analyze cross-schema relationships
- **Performance Outliers**: Identify models with unusual execution patterns

## Quick Start

### 1. Setup

```bash
# Clone or create the project directory
cd /Users/marquein/knowledge_graph

# Install dependencies
pip install -r requirements.txt

# Configure Snowflake connection
cp config/env_example.txt .env
# Edit .env with your Snowflake credentials
```

### 2. Prerequisites

Ensure you have:
- A configured `~/.dbt/profiles.yml` file with a Snowflake connection
- dbt Elementary package installed and run in your dbt project
- Elementary tables populated in your Snowflake database

The system will automatically read your dbt profiles and use the `dev` target by default.

### 3. Basic Usage

```python
from src.data_extractor import ElementaryDataExtractor
from src.graph_builder import KnowledgeGraphBuilder
from src.graph_storage import GraphStorageManager
from src.graph_analyzer import GraphAnalyzer

# Extract data from Snowflake (uses ~/.dbt/profiles.yml automatically)
extractor = ElementaryDataExtractor(target='dev')  # or your preferred target
data = extractor.extract_all_data()

# Build knowledge graph
builder = KnowledgeGraphBuilder()
graph = builder.build_graph_from_data(data)

# Save graph
storage = GraphStorageManager(backend_type='filesystem')
storage.save_graph(graph)

# Analyze graph
analyzer = GraphAnalyzer(graph)
coverage = analyzer.get_test_coverage()
performance = analyzer.get_performance_analysis()
```

### 4. Check Your dbt Configuration

```bash
# Check your dbt profiles and Snowflake connections
python scripts/check_dbt_profiles.py
```

This will show you:
- Available dbt profiles and targets
- Snowflake connection details
- Whether Elementary tables exist
- Recommended configuration

### 5. Run Examples

```bash
# Basic usage example
python examples/basic_usage.py

# Advanced queries example
python examples/advanced_queries.py
```

## Project Structure

```
knowledge_graph/
├── src/                          # Core modules
│   ├── data_extractor.py         # Snowflake data extraction
│   ├── graph_entities.py         # Graph node/edge definitions
│   ├── graph_builder.py          # Graph construction logic
│   ├── graph_storage.py          # Storage backends (filesystem, Neo4j)
│   └── graph_analyzer.py         # Analysis and query functions
├── config/                       # Configuration
│   ├── database_config.py        # dbt profiles.yml reader
│   └── env_example.txt          # Environment variables template (legacy)
├── scripts/                      # Utility scripts
│   └── check_dbt_profiles.py    # dbt configuration checker
├── examples/                     # Usage examples
│   ├── basic_usage.py           # Getting started example
│   └── advanced_queries.py     # Complex analysis examples
├── data/                        # Stored graph data
├── notebooks/                   # Jupyter analysis notebooks
└── requirements.txt            # Python dependencies
```

## Elementary Tables Used

The system extracts data from these Elementary tables in Snowflake:

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `dbt_models` | Model metadata | `unique_id`, `name`, `depends_on_nodes`, `materialization` |
| `dbt_tests` | Test definitions | `unique_id`, `name`, `depends_on_nodes`, `test_type` |
| `dbt_sources` | Source table info | `unique_id`, `source_name`, `table_name` |
| `elementary_test_results` | Test execution results | `test_unique_id`, `status`, `result_rows`, `execution_time` |
| `dbt_run_results` | Model execution results | `model_unique_id`, `status`, `execution_time`, `rows_affected` |
| `dbt_invocations` | dbt run metadata | `invocation_id`, `job_name`, `command`, `dbt_version` |
| `model_columns` | Column-level metadata | `model_unique_id`, `column_name`, `data_type` |

## Graph Schema

### Node Types
- **Model**: dbt models (tables/views)
- **Test**: Data quality tests
- **Source**: Source tables
- **Column**: Model columns
- **Invocation**: dbt execution runs

### Edge Types
- **depends_on**: Model dependencies
- **has_column**: Model to column relationships
- **tests**: Test to model relationships
- **executed_in**: Invocation to model/test relationships
- **has_result**: Invocation to test results

## Analysis Examples

### Lineage Analysis
```python
# Get full lineage for a model
lineage = analyzer.get_model_lineage('model.project.customer_orders')
print(f"Upstream: {lineage['upstream']}")
print(f"Downstream: {lineage['downstream']}")
```

### Test Coverage
```python
# Analyze test coverage
coverage = analyzer.get_test_coverage()
print(f"Coverage: {coverage['coverage_percentage']:.1f}%")
print(f"Models without tests: {len(coverage['models_without_tests'])}")
```

### Performance Analysis
```python
# Find slow models
performance = analyzer.get_performance_analysis()
slowest = performance['model_performance'][0]
print(f"Slowest model: {slowest['model_name']} ({slowest['avg_execution_time']:.2f}s)")
```

### Impact Analysis
```python
# Analyze impact of changes
impact = analyzer.get_impact_analysis('model.project.dim_customers')
print(f"Total affected models: {impact['total_affected']}")
```

## Storage Options

### File System Storage
```python
storage = GraphStorageManager(backend_type='filesystem')
storage.save_graph(graph)
```

### Neo4j Storage
```python
storage = GraphStorageManager(
    backend_type='neo4j',
    config={'uri': 'bolt://localhost:7687', 'user': 'neo4j', 'password': 'password'}
)
storage.save_graph(graph)
```

## Configuration Options

### Using Different dbt Profiles/Targets

```python
# Use a specific profile and target
extractor = ElementaryDataExtractor(
    profile_name='my_project', 
    target='prod'
)

# Auto-detect Snowflake profile with custom target
extractor = ElementaryDataExtractor(target='staging')

# Manually specify elementary schema
extractor = ElementaryDataExtractor()
extractor.set_elementary_schema('my_schema_elementary')
```

### Manual Connection Configuration

```python
# Override with manual config (bypasses dbt profiles)
manual_config = {
    'user': 'my_user',
    'password': 'my_password',
    'account': 'my_account.region',
    'warehouse': 'COMPUTE_WH',
    'database': 'ANALYTICS',
    'schema': 'DBT_ELEMENTARY'
}

extractor = ElementaryDataExtractor(connection_config=manual_config)
```

## Advanced Queries

The project includes advanced query capabilities:

- **Tag-based filtering**: Find models by tags
- **Schema analysis**: Cross-schema dependencies
- **Performance outliers**: Models with unusual execution patterns
- **Dependency chains**: Longest model dependency paths
- **Data lineage**: Trace models back to source tables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Requirements

- Python 3.8+
- Snowflake access with Elementary tables
- Optional: Neo4j for graph database storage

## License

This project is open source. See LICENSE file for details.