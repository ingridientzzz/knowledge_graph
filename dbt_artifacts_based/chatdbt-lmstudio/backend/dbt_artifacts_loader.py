import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from llama_index.core.schema import Document
from llama_index.core.graph_stores.types import (
    EntityNode, 
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY
)


class DBTArtifactsLoader:
    """
    Loads and processes dbt artifacts (manifest.json, catalog.json, graph_summary.json)
    to create PropertyGraph data structures for LlamaIndex.
    """
    
    def __init__(self, artifacts_path: str):
        """
        Initialize the loader with dbt artifacts directory path.
        
        Args:
            artifacts_path: Path to directory containing dbt artifacts
        """
        self.artifacts_path = Path(artifacts_path)
        self.manifest_file = self.artifacts_path / "manifest.json"
        self.catalog_file = self.artifacts_path / "catalog.json"  
        self.graph_summary_file = self.artifacts_path / "graph_summary.json"
        
        # Validate required files exist
        for file_path in [self.manifest_file, self.graph_summary_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required dbt artifact not found: {file_path}")
        
        self._manifest_data = None
        self._catalog_data = None
        self._graph_data = None
        
    def _load_artifacts(self):
        """Load all artifact files if not already loaded."""
        if self._manifest_data is None:
            print("Loading manifest.json...")
            with open(self.manifest_file, 'r', encoding='utf-8') as f:
                self._manifest_data = json.load(f)
            print(f"Loaded manifest with {len(self._manifest_data.get('nodes', {}))} nodes")
            
        if self._graph_data is None:
            print("Loading graph_summary.json...")
            with open(self.graph_summary_file, 'r', encoding='utf-8') as f:
                self._graph_data = json.load(f)
            print(f"Loaded graph summary with {len(self._graph_data.get('linked', {}))} linked nodes")
            
        if self._catalog_data is None and self.catalog_file.exists():
            print("Loading catalog.json...")
            with open(self.catalog_file, 'r', encoding='utf-8') as f:
                self._catalog_data = json.load(f)
            print(f"Loaded catalog with {len(self._catalog_data.get('nodes', {}))} cataloged nodes")

    def _extract_entity_nodes(self) -> List[EntityNode]:
        """Extract entity nodes from dbt artifacts."""
        self._load_artifacts()
        entity_nodes = []
        
        # Process manifest nodes
        manifest_nodes = self._manifest_data.get('nodes', {})
        catalog_nodes = self._catalog_data.get('nodes', {}) if self._catalog_data else {}
        
        for node_id, node_data in manifest_nodes.items():
            # Get catalog data if available
            catalog_info = catalog_nodes.get(node_id, {})
            
            # Extract node properties
            properties = {
                'id': node_id,
                'name': node_data.get('name', ''),
                'resource_type': node_data.get('resource_type', ''),
                'database': node_data.get('database', ''),
                'schema': node_data.get('schema', ''),
                'package_name': node_data.get('package_name', ''),
                'description': node_data.get('description', ''),
                'path': node_data.get('original_file_path', ''),
                'tags': node_data.get('tags', []),
                'meta': node_data.get('meta', {}),
                'config': node_data.get('config', {}),
                'columns': self._extract_column_info(node_data, catalog_info),
                'depends_on': node_data.get('depends_on', {}),
            }
            
            # Add catalog metadata if available
            if catalog_info:
                properties['table_metadata'] = catalog_info.get('metadata', {})
                properties['column_count'] = len(catalog_info.get('columns', {}))
            
            # Create entity node
            entity = EntityNode(
                name=node_id,
                label=node_data.get('resource_type', 'node'),
                properties=properties
            )
            entity_nodes.append(entity)
            
        # Process sources from manifest
        sources = self._manifest_data.get('sources', {})
        for source_id, source_data in sources.items():
            properties = {
                'id': source_id,
                'name': source_data.get('name', ''),
                'resource_type': 'source',
                'database': source_data.get('database', ''),
                'schema': source_data.get('schema', ''),
                'package_name': source_data.get('package_name', ''),
                'source_name': source_data.get('source_name', ''),
                'description': source_data.get('description', ''),
                'tags': source_data.get('tags', []),
                'meta': source_data.get('meta', {}),
                'columns': self._extract_column_info(source_data, {}),
                'freshness': source_data.get('freshness', {}),
            }
            
            entity = EntityNode(
                name=source_id,
                label='source',
                properties=properties
            )
            entity_nodes.append(entity)
            
        print(f"Extracted {len(entity_nodes)} entity nodes")
        return entity_nodes

    def _extract_column_info(self, node_data: Dict, catalog_info: Dict) -> List[Dict]:
        """Extract column information from node and catalog data."""
        columns = []
        
        # Get column info from manifest
        manifest_columns = node_data.get('columns', {})
        catalog_columns = catalog_info.get('columns', {})
        
        for col_name, col_data in manifest_columns.items():
            column_info = {
                'name': col_name,
                'description': col_data.get('description', ''),
                'data_type': col_data.get('data_type', ''),
                'meta': col_data.get('meta', {}),
                'tags': col_data.get('tags', []),
            }
            
            # Add catalog info if available
            if col_name in catalog_columns:
                catalog_col = catalog_columns[col_name]
                column_info.update({
                    'catalog_type': catalog_col.get('type', ''),
                    'index': catalog_col.get('index', 0),
                    'comment': catalog_col.get('comment', ''),
                })
            
            columns.append(column_info)
            
        return columns

    def _extract_relations(self) -> List[Relation]:
        """Extract relations between nodes from dbt artifacts."""
        relations = []
        
        # Extract dependencies from manifest
        manifest_nodes = self._manifest_data.get('nodes', {})
        for node_id, node_data in manifest_nodes.items():
            depends_on = node_data.get('depends_on', {})
            
            # Process node dependencies
            for dep_node in depends_on.get('nodes', []):
                relation = Relation(
                    label='depends_on',
                    source_id=node_id,
                    target_id=dep_node,
                    properties={
                        'relationship_type': 'dependency',
                        'source_resource_type': node_data.get('resource_type', ''),
                        'created_from': 'manifest_depends_on'
                    }
                )
                relations.append(relation)
        
        # Extract relationships from graph summary 
        graph_linked = self._graph_data.get('linked', {})
        for node_key, node_info in graph_linked.items():
            node_name = node_info.get('name', '')
            node_type = node_info.get('type', '')
            successors = node_info.get('succ', [])
            
            # Create successor relationships
            for succ_key in successors:
                if str(succ_key) in graph_linked:
                    succ_info = graph_linked[str(succ_key)]
                    succ_name = succ_info.get('name', '')
                    
                    relation = Relation(
                        label='flows_to',
                        source_id=node_name,
                        target_id=succ_name,
                        properties={
                            'relationship_type': 'data_flow',
                            'source_type': node_type,
                            'target_type': succ_info.get('type', ''),
                            'created_from': 'graph_summary'
                        }
                    )
                    relations.append(relation)
        
        print(f"Extracted {len(relations)} relations")
        return relations

    def _create_property_graph_documents(self, entity_nodes: List[EntityNode], relations: List[Relation]) -> List[Document]:
        """Create documents that contain property graph data for indexing."""
        documents = []
        
        # First, create a comprehensive package statistics document
        package_stats = self._calculate_package_statistics(entity_nodes)
        stats_content = self._create_package_statistics_content(package_stats)
        
        stats_doc = Document(
            text=stats_content,
            metadata={
                'document_type': 'package_statistics',
                'source': 'dbt_artifacts',
                'contains_package_stats': True,
                'total_packages': len(package_stats),
                'keywords': 'package statistics, package rankings, packages with most models, packages with most tests',
                'description': f'Statistical analysis of all {len(package_stats)} dbt packages including rankings by models and tests',
            }
        )
        documents.append(stats_doc)
        
        # Create a dedicated complete package listing document for easy retrieval
        complete_package_list = self._create_complete_package_listing(package_stats)
        complete_list_doc = Document(
            text=complete_package_list,
            metadata={
                'document_type': 'complete_package_listing',
                'source': 'dbt_artifacts',
                'total_packages': len(package_stats),
                'name': 'all_packages_complete_list',
                'is_complete_package_list': True,
                'keywords': 'list all packages, show all packages, package names, package directory, complete package list',
                'description': f'Complete listing of all {len(package_stats)} dbt packages with statistics',
            }
        )
        documents.append(complete_list_doc)
        
        # Create comprehensive package model directories for ALL packages with models
        for package_name, stats in package_stats.items():
            if stats.get('model', 0) > 0:  # Only create docs for packages with models
                # Create comprehensive model directory for each package
                package_models = self._get_package_models(package_name, entity_nodes)
                if package_models:
                    models_content = self._create_package_models_directory(package_name, package_models)
                    
                    models_doc = Document(
                        text=models_content,
                        metadata={
                            'document_type': 'package_models_directory',
                            'node_type': 'package_models_directory',
                            'package_name': package_name,
                            'source': 'dbt_artifacts',
                            'model_count': len(package_models),
                            'name': f'{package_name}_models_directory',
                            'is_complete_model_list': True
                        }
                    )
                    documents.append(models_doc)
        
        # Create detailed package documents for ALL packages (let LLM handle pagination)
        for package_name, stats in sorted(package_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            package_content = self._create_package_detail_content(package_name, stats, entity_nodes)
            
            package_doc = Document(
                text=package_content,
                metadata={
                    'document_type': 'package_detail',
                    'package_name': package_name,
                    'source': 'dbt_artifacts',
                    'model_count': stats.get('model', 0),
                    'test_count': stats.get('test', 0),
                    'source_count': stats.get('source', 0),
                    'seed_count': stats.get('seed', 0),
                    'keywords': f'{package_name} package, {package_name} sources, {package_name} models, sources in {package_name}',
                    'description': f'Complete details for {package_name} package including all {stats.get("source", 0)} sources and {stats.get("model", 0)} models',
                }
            )
            documents.append(package_doc)
        
        # Create summary documents for different node types
        node_types = {}
        for entity in entity_nodes:
            resource_type = entity.properties.get('resource_type', 'unknown')
            if resource_type not in node_types:
                node_types[resource_type] = []
            node_types[resource_type].append(entity)
        
        # Create documents for each node type
        for resource_type, nodes in node_types.items():
            content_parts = [
                f"# DBT {resource_type.upper()} Resources\n",
                f"This document contains information about {len(nodes)} {resource_type} resources in the dbt project.\n\n"
            ]
            
            # Show more entities for important resource types, fewer for others
            limit = 20 if resource_type in ['model', 'test'] else 10
            displayed_nodes = nodes[:limit] if len(nodes) > limit else nodes
            truncated = len(nodes) > limit
            
            for entity in displayed_nodes:
                props = entity.properties
                content_parts.append(f"## {props.get('name', entity.name)}\n")
                content_parts.append(f"- **ID**: {entity.name}\n")
                content_parts.append(f"- **Database**: {props.get('database', 'N/A')}\n")
                content_parts.append(f"- **Schema**: {props.get('schema', 'N/A')}\n")
                content_parts.append(f"- **Package**: {props.get('package_name', 'N/A')}\n")
                
                if props.get('description'):
                    content_parts.append(f"- **Description**: {props['description']}\n")
                
                # Add column information
                columns = props.get('columns', [])
                if columns:
                    content_parts.append(f"- **Columns** ({len(columns)}):\n")
                    for col in columns[:5]:  # Show first 5 columns
                        content_parts.append(f"  - {col['name']}: {col.get('data_type', 'unknown')} - {col.get('description', 'No description')}\n")
                
                content_parts.append("\n")
            
            # Add truncation notice if needed
            if truncated:
                remaining = len(nodes) - limit
                content_parts.append(f"*Note: Showing {limit} of {len(nodes)} {resource_type} resources. {remaining} additional {resource_type}s are available but not displayed here for brevity.*\n\n")
            
            # Get relevant relations for this document's nodes (limited)
            node_names = [entity.name for entity in nodes]
            relevant_relations = [rel for rel in relations if 
                                rel.source_id in node_names or rel.target_id in node_names]
            
            doc = Document(
                text="".join(content_parts),
                metadata={
                    'document_type': f'dbt_{resource_type}_summary',
                    'resource_type': resource_type,
                    'node_count': len(nodes),
                    'source': 'dbt_artifacts',
                    'entity_names': [entity.name for entity in nodes][:20],  # Only first 20 names
                    'relation_count': len(relevant_relations),
                    # Store only essential data, not full objects to avoid metadata size issues
                    'sample_entities': [{'name': e.name, 'type': e.label} for e in nodes][:5],
                    'sample_relations': [{'source': r.source_id, 'target': r.target_id, 'type': r.label} 
                                       for r in relevant_relations][:10]
                }
            )
            documents.append(doc)
        
        # Create a relationships overview document
        relationship_types = {}
        for rel in relations:
            rel_type = rel.properties.get('relationship_type', 'unknown')
            if rel_type not in relationship_types:
                relationship_types[rel_type] = []
            relationship_types[rel_type].append(rel)
        
        rel_content = ["# DBT Project Relationships\n\n"]
        for rel_type, rels in relationship_types.items():
            rel_content.append(f"## {rel_type.replace('_', ' ').title()} ({len(rels)} relationships)\n")
            for rel in rels[:10]:  # Show sample relationships
                rel_content.append(f"- {rel.source_id} → {rel.target_id}\n")
            rel_content.append("\n")
        
        rel_doc = Document(
            text="".join(rel_content),
            metadata={
                'document_type': 'dbt_relationships_summary',
                'relationship_count': len(relations),
                'source': 'dbt_artifacts',
                'relationship_types': list(relationship_types.keys()),
                'sample_relationships': [{'source': r.source_id, 'target': r.target_id, 'type': r.label} 
                                       for r in relations][:20]  # Only first 20 relationships
            }
        )
        documents.append(rel_doc)
        
        return documents

    def _calculate_package_statistics(self, entity_nodes: List[EntityNode]) -> Dict[str, Dict[str, int]]:
        """Calculate statistics for each package."""
        package_stats = {}
        
        for entity in entity_nodes:
            props = entity.properties
            package = props.get('package_name', 'unknown')
            resource_type = props.get('resource_type', 'unknown')
            
            if package not in package_stats:
                package_stats[package] = {'model': 0, 'test': 0, 'source': 0, 'seed': 0, 'total': 0}
            
            if resource_type in package_stats[package]:
                package_stats[package][resource_type] += 1
            package_stats[package]['total'] += 1
        
        return package_stats

    def _create_package_statistics_content(self, package_stats: Dict[str, Dict[str, int]]) -> str:
        """Create content for package statistics document."""
        content_parts = [
            "# DBT Package Statistics Report\n\n",
            "This document contains comprehensive statistics about all dbt packages in the project.\n\n",
            "## Package Rankings by Total Resources\n\n"
        ]
        
        sorted_packages = sorted(package_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        content_parts.append("| Rank | Package Name | Models | Tests | Sources | Seeds | Total |\n")
        content_parts.append("|------|-------------|---------|-------|---------|-------|-------|\n")
        
        for i, (package, stats) in enumerate(sorted_packages, 1):
            models = stats.get('model', 0)
            tests = stats.get('test', 0)
            sources = stats.get('source', 0)
            seeds = stats.get('seed', 0)
            total = stats['total']
            
            content_parts.append(f"| {i} | **{package}** | {models} | {tests} | {sources} | {seeds} | {total} |\n")
        
        content_parts.append(f"\n\n## Summary Statistics\n\n")
        content_parts.append(f"- **Total packages**: {len(package_stats)}\n")
        
        total_models = sum(stats.get('model', 0) for stats in package_stats.values())
        total_tests = sum(stats.get('test', 0) for stats in package_stats.values())
        total_sources = sum(stats.get('source', 0) for stats in package_stats.values())
        total_seeds = sum(stats.get('seed', 0) for stats in package_stats.values())
        
        content_parts.append(f"- **Total models**: {total_models}\n")
        content_parts.append(f"- **Total tests**: {total_tests}\n")
        content_parts.append(f"- **Total sources**: {total_sources}\n")
        content_parts.append(f"- **Total seeds**: {total_seeds}\n")
        
        # Add a comprehensive list of ALL packages for easy querying
        content_parts.append(f"\n\n## Complete Package List\n\n")
        content_parts.append("All packages in this dbt project:\n\n")
        for i, (package, stats) in enumerate(sorted_packages, 1):
            total = stats['total']
            models = stats.get('model', 0)
            tests = stats.get('test', 0)
            content_parts.append(f"{i:2d}. **{package}** ({total} resources: {models} models, {tests} tests)\n")
        
        return "".join(content_parts)

    def _create_complete_package_listing(self, package_stats: Dict[str, Dict[str, int]]) -> str:
        """Create a dedicated document listing all packages with their statistics."""
        content_parts = [
            "# Complete Package Directory - ALL DBT PACKAGES\n\n",
            "COMPLETE LIST OF ALL PACKAGES - COMPREHENSIVE PACKAGE DIRECTORY\n\n",
            "This document contains a COMPLETE and COMPREHENSIVE listing of ALL dbt packages in the project.\n",
            f"**TOTAL PACKAGES: {len(package_stats)}** (Every single package is listed below)\n\n",
            "KEYWORDS: list all packages, show all packages, complete package list, all package names, package directory, comprehensive package listing\n\n"
        ]
        
        sorted_packages = sorted(package_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        content_parts.append("## Quick Reference: All Package Names\n\n")
        package_names = [pkg for pkg, _ in sorted_packages]
        content_parts.append(f"ALL {len(package_names)} PACKAGE NAMES: ")
        content_parts.append(", ".join(package_names))
        content_parts.append("\n\n")
        
        content_parts.append("## All Packages Ranked by Total Resources\n\n")
        
        for i, (package, stats) in enumerate(sorted_packages, 1):
            models = stats.get('model', 0)
            tests = stats.get('test', 0)
            sources = stats.get('source', 0)
            seeds = stats.get('seed', 0)
            total = stats['total']
            
            content_parts.append(f"**{i:2d}. {package}**\n")
            content_parts.append(f"   - Models: {models}\n")
            content_parts.append(f"   - Tests: {tests}\n")
            content_parts.append(f"   - Sources: {sources}\n")
            content_parts.append(f"   - Seeds: {seeds}\n")
            content_parts.append(f"   - **Total: {total}**\n\n")
        
        # Add searchable keywords and guidance for LLM
        content_parts.append("## Search Keywords\n\n")
        content_parts.append("Package names: " + ", ".join([pkg for pkg, _ in sorted_packages]) + "\n\n")
        content_parts.append("## Usage Instructions for AI Assistant\n\n")
        content_parts.append("This document answers questions like:\n")
        content_parts.append("- What packages have the most models?\n")
        content_parts.append("- What packages have the most tests?\n") 
        content_parts.append("- List all packages in the project\n")
        content_parts.append("- How many packages are there?\n")
        content_parts.append("- Package statistics and rankings\n\n")
        content_parts.append("**Note for AI Assistant**: When displaying package lists to users, you can:\n")
        content_parts.append("1. Show the top 10-15 packages initially\n")
        content_parts.append("2. Ask if the user wants to see more packages\n")
        content_parts.append("3. Continue showing additional packages in batches\n")
        content_parts.append("4. Always mention the total count available\n")
        
        return "".join(content_parts)

    def _create_package_detail_content(self, package_name: str, stats: Dict[str, int], entity_nodes: List[EntityNode]) -> str:
        """Create detailed content for a specific package."""
        content_parts = [
            f"# {package_name} Package Details\n\n",
            f"**SEARCH KEYWORDS**: {package_name} package, {package_name} sources, {package_name} models, sources in {package_name}, what sources does {package_name} use\n\n",
            f"Comprehensive information about the **{package_name}** dbt package.\n\n",
            f"## Package Statistics\n\n",
            f"- **Models**: {stats.get('model', 0)}\n",
            f"- **Tests**: {stats.get('test', 0)}\n",
            f"- **Sources**: {stats.get('source', 0)}\n",
            f"- **Seeds**: {stats.get('seed', 0)}\n",
            f"- **Total Resources**: {stats['total']}\n\n",
        ]
        
        # Add models in this package
        package_entities = [e for e in entity_nodes if e.properties.get('package_name') == package_name]
        
        models = [e for e in package_entities if e.properties.get('resource_type') == 'model']
        if models:
            content_parts.append(f"## Models in {package_name} ({len(models)} models)\n\n")
            # Show ALL models for package queries - users expect complete lists
            for model in models:
                name = model.properties.get('name', model.name)
                description = model.properties.get('description', 'No description')
                # Truncate very long descriptions to keep document manageable
                truncated_desc = description[:80] + '...' if len(description) > 80 else description
                content_parts.append(f"- **{name}**: {truncated_desc}\n")
            content_parts.append("\n")
        
        tests = [e for e in package_entities if e.properties.get('resource_type') == 'test']
        if tests:
            content_parts.append(f"## Tests in {package_name} ({len(tests)} tests)\n\n")
            
            # Group tests by type for better organization
            test_types = {}
            for test in tests:
                props = test.properties
                test_name = props.get('name', 'unknown')
                # Extract test type from name patterns
                if 'unique' in test_name.lower():
                    test_type = 'Uniqueness Tests'
                elif 'not_null' in test_name.lower():
                    test_type = 'Not Null Tests'
                elif 'accepted_values' in test_name.lower():
                    test_type = 'Accepted Values Tests'
                elif 'relationships' in test_name.lower():
                    test_type = 'Relationship Tests'
                elif 'custom' in test_name.lower() or 'generic' in test_name.lower():
                    test_type = 'Custom Tests'
                else:
                    test_type = 'Other Tests'
                
                if test_type not in test_types:
                    test_types[test_type] = []
                test_types[test_type].append(test)
            
            # Show tests by category with details
            for test_type, type_tests in sorted(test_types.items()):
                content_parts.append(f"### {test_type} ({len(type_tests)})\n")
                
                # Show first 10 tests in each category with truncation notice
                displayed_tests = type_tests[:10]
                for test in displayed_tests:
                    props = test.properties
                    test_name = props.get('name', 'unknown')
                    description = props.get('description', '')
                    
                    content_parts.append(f"- **{test_name}**\n")
                    if description:
                        truncated_desc = description[:80] + '...' if len(description) > 80 else description
                        content_parts.append(f"  - {truncated_desc}\n")
                
                if len(type_tests) > 10:
                    remaining = len(type_tests) - 10
                    content_parts.append(f"\n*...and {remaining} more {test_type.lower()}*\n")
                
                content_parts.append("\n")
            
            # Add summary
            content_parts.append(f"**Test Summary**: {len(tests)} total data quality tests ensuring data integrity and business rules.\n\n")
        
        sources = [e for e in package_entities if e.properties.get('resource_type') == 'source']
        if sources:
            content_parts.append(f"## Sources in {package_name} ({len(sources)} sources)\n\n")
            # Show ALL sources with detailed information (like we do for models)
            for source in sources:
                props = source.properties
                name = props.get('name', 'unknown')
                source_name = props.get('source_name', 'unknown')
                database = props.get('database', 'N/A')
                schema = props.get('schema', 'N/A')
                description = props.get('description', 'No description')
                freshness = props.get('freshness', {})
                
                # Format the source entry with comprehensive information
                content_parts.append(f"- **{source_name}.{name}**\n")
                content_parts.append(f"  - Location: {database}.{schema}.{props.get('identifier', name)}\n")
                
                if description and description != 'No description':
                    truncated_desc = description[:100] + '...' if len(description) > 100 else description
                    content_parts.append(f"  - Description: {truncated_desc}\n")
                
                # Add freshness information if available
                if freshness:
                    warn = freshness.get('warn_after', {})
                    error = freshness.get('error_after', {})
                    freshness_info = []
                    
                    if warn and warn.get('count') and warn.get('period'):
                        freshness_info.append(f"warn: {warn['count']} {warn['period']}")
                    if error and error.get('count') and error.get('period'):
                        freshness_info.append(f"error: {error['count']} {error['period']}")
                    
                    if freshness_info:
                        content_parts.append(f"  - Freshness: {', '.join(freshness_info)}\n")
                    elif freshness:  # Has freshness config but no valid thresholds
                        content_parts.append(f"  - Freshness: configured (no thresholds)\n")
                
                # Add column count if available
                columns = props.get('columns', [])
                if columns:
                    content_parts.append(f"  - Columns: {len(columns)}\n")
                
                content_parts.append("\n")
            
            content_parts.append("\n")
        
        seeds = [e for e in package_entities if e.properties.get('resource_type') == 'seed']
        if seeds:
            content_parts.append(f"## Seeds in {package_name} ({len(seeds)} seeds)\n\n")
            content_parts.append("Static data files loaded into the warehouse:\n\n")
            
            for seed in seeds:
                props = seed.properties
                name = props.get('name', 'unknown')
                description = props.get('description', '')
                path = props.get('path', '')
                columns = props.get('columns', [])
                
                content_parts.append(f"- **{name}**\n")
                
                if description:
                    truncated_desc = description[:100] + '...' if len(description) > 100 else description
                    content_parts.append(f"  - Description: {truncated_desc}\n")
                
                if path:
                    content_parts.append(f"  - File: {path}\n")
                
                if columns:
                    content_parts.append(f"  - Columns: {len(columns)}\n")
                    # Show first few column names
                    col_names = [col.get('name', 'unknown') for col in columns[:5]]
                    if col_names:
                        content_parts.append(f"  - Sample columns: {', '.join(col_names)}\n")
                        if len(columns) > 5:
                            content_parts.append(f"    (and {len(columns)-5} more columns)\n")
                
                content_parts.append("\n")
            
            content_parts.append("\n")
        
        # Handle other resource types that might exist
        other_resources = [e for e in package_entities if e.properties.get('resource_type') not in ['model', 'test', 'source', 'seed']]
        if other_resources:
            # Group by resource type
            other_types = {}
            for resource in other_resources:
                rt = resource.properties.get('resource_type', 'unknown')
                if rt not in other_types:
                    other_types[rt] = []
                other_types[rt].append(resource)
            
            for resource_type, resources in sorted(other_types.items()):
                content_parts.append(f"## {resource_type.title()}s in {package_name} ({len(resources)} {resource_type}s)\n\n")
                
                for resource in resources:
                    props = resource.properties
                    name = props.get('name', 'unknown')
                    description = props.get('description', '')
                    path = props.get('path', '')
                    
                    content_parts.append(f"- **{name}**\n")
                    
                    if description:
                        truncated_desc = description[:100] + '...' if len(description) > 100 else description
                        content_parts.append(f"  - Description: {truncated_desc}\n")
                    
                    if path:
                        content_parts.append(f"  - File: {path}\n")
                    
                    # Add resource-type specific information
                    if resource_type == 'operation':
                        content_parts.append(f"  - Type: Database operation/hook\n")
                    elif resource_type == 'snapshot':
                        strategy = props.get('config', {}).get('strategy', 'N/A')
                        content_parts.append(f"  - Strategy: {strategy}\n")
                    elif resource_type == 'analysis':
                        content_parts.append(f"  - Type: Analysis query (not materialized)\n")
                    
                    content_parts.append("\n")
                
                content_parts.append("\n")
        
        return "".join(content_parts)

    def _get_package_models(self, package_name: str, entity_nodes: List[EntityNode]) -> List[Dict[str, str]]:
        """Get all models for a specific package."""
        models = []
        for entity in entity_nodes:
            if (entity.properties.get('package_name') == package_name and 
                entity.properties.get('resource_type') == 'model'):
                models.append({
                    'name': entity.properties.get('name', 'unknown'),
                    'description': entity.properties.get('description', 'No description available'),
                    'id': entity.properties.get('id', 'unknown')
                })
        return models

    def _create_package_models_directory(self, package_name: str, models: List[Dict[str, str]]) -> str:
        """Create comprehensive model directory content for a package."""
        content_parts = [
            f"PACKAGE: {package_name}",
            f"Total Models: {len(models)}",
            "",
            "ALL MODELS IN PACKAGE:"
        ]
        
        # List ALL models with their descriptions
        for model in models:
            model_name = model['name']
            model_desc = model['description']
            content_parts.append(f"• {model_name}: {model_desc}")
        
        return "\n".join(content_parts)

    def load_documents(self) -> List[Document]:
        """
        Load dbt artifacts and create documents for VectorStoreIndex.
        
        Returns:
            List of Document objects for vector indexing
        """
        print("Loading dbt artifacts for document creation...")
        self._load_artifacts()
        
        # Extract graph components (we still need these for comprehensive documents)
        entity_nodes = self._extract_entity_nodes()
        relations = self._extract_relations()
        
        # Create documents for indexing
        documents = self._create_property_graph_documents(entity_nodes, relations)
        
        print(f"Created {len(documents)} documents from dbt artifacts")
        
        return documents

    def load_property_graph_data(self) -> Tuple[List[Document], List[EntityNode], List[Relation]]:
        """
        Load dbt artifacts and create property graph data.
        
        Returns:
            Tuple of (documents, entity_nodes, relations)
        """
        print("Loading dbt artifacts for property graph...")
        self._load_artifacts()
        
        # Extract graph components
        entity_nodes = self._extract_entity_nodes()
        relations = self._extract_relations()
        
        # Create documents for indexing
        documents = self._create_property_graph_documents(entity_nodes, relations)
        
        print(f"Created {len(documents)} documents with property graph data")
        print(f"- {len(entity_nodes)} entity nodes")
        print(f"- {len(relations)} relations")
        
        return documents, entity_nodes, relations
