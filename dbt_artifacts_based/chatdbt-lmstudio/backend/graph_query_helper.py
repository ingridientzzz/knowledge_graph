#!/usr/bin/env python3
"""
Graph Query Helper - Direct access to relationship data
Best method for accurate dependency queries
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import time

class GraphQueryHelper:
    """Fast, accurate graph relationship queries."""
    
    def __init__(self, storage_path: str = "./storage"):
        self.storage_path = Path(storage_path)
        self.graph_store_path = self.storage_path / "graph_store.json"
        self._graph_data = None
        self._load_time = None
    
    def _load_graph(self):
        """Load graph data if not already loaded."""
        if self._graph_data is None:
            start_time = time.time()
            with open(self.graph_store_path, 'r') as f:
                self._graph_data = json.load(f)
            self._load_time = time.time() - start_time
            print(f"📊 Graph loaded in {self._load_time:.3f}s")
    
    def find_dependencies(self, model_name: str, include_external: bool = True) -> Dict[str, Any]:
        """Find what depends on a specific model (downstream dependencies)."""
        self._load_graph()
        
        graph_dict = self._graph_data['graph_dict']
        relations = graph_dict['relations']
        
        # Direct dependencies (same package)
        direct_deps = []
        # External dependencies (cross-package)
        external_deps = []
        
        model_name_lower = model_name.lower()
        
        for rel in relations:
            source = rel.get('source', '')
            target = rel.get('target', '')
            label = rel.get('label', 'unknown')
            
            # For "depends_on" relationships: if our model is the target, then source depends on us
            # For "flows_to" relationships: if our model is the source, then target depends on us
            depends_on_us = False
            dependent_model = None
            
            if label == 'depends_on' and model_name_lower in target.lower():
                # Source depends on our model (target)
                depends_on_us = True
                dependent_model = source
            elif label == 'flows_to' and model_name_lower in source.lower():
                # Our model flows to target, so target depends on us
                depends_on_us = True
                dependent_model = target
            
            if depends_on_us and dependent_model:
                # Determine if this is same package or external
                is_external = not any(pkg in dependent_model.lower() for pkg in model_name_lower.split('_'))
                
                dep_info = {
                    'dependent_model': dependent_model,
                    'depends_on': f"model.{model_name}" if not model_name.startswith('model.') else model_name,
                    'relationship_type': label,
                    'is_external': is_external
                }
                
                if is_external:
                    external_deps.append(dep_info)
                else:
                    direct_deps.append(dep_info)
        
        return {
            'query': f"What depends on {model_name}",
            'direct_dependencies': direct_deps,
            'external_dependencies': external_deps if include_external else [],
            'total_count': len(direct_deps) + len(external_deps),
            'summary': f"Found {len(direct_deps)} direct + {len(external_deps)} external dependencies"
        }
    
    def find_upstream(self, model_name: str) -> Dict[str, Any]:
        """Find what a model depends on (upstream dependencies)."""
        self._load_graph()
        
        graph_dict = self._graph_data['graph_dict']
        relations = graph_dict['relations']
        
        upstream_deps = []
        model_name_lower = model_name.lower()
        
        for rel in relations:
            source = rel.get('source', '')
            target = rel.get('target', '')
            
            # Check if our model is the source (it depends on target)
            if model_name_lower in source.lower():
                upstream_deps.append({
                    'model': source,
                    'depends_on': target,
                    'relationship_type': rel.get('label', 'unknown')
                })
        
        return {
            'query': f"What {model_name} depends on",
            'upstream_dependencies': upstream_deps,
            'count': len(upstream_deps)
        }
    
    def search_models(self, search_term: str) -> List[str]:
        """Search for models containing a term."""
        self._load_graph()
        
        graph_dict = self._graph_data['graph_dict']
        entities = graph_dict['entities']
        
        matching_models = []
        search_term_lower = search_term.lower()
        
        for entity_id, entity_data in entities.items():
            if search_term_lower in entity_id.lower():
                matching_models.append(entity_id)
        
        return sorted(matching_models)
    
    def find_commonalities(self, entities: List[str]) -> Dict[str, Any]:
        """Find commonalities between multiple dbt entities (models, sources, etc.)."""
        self._load_graph()
        
        graph_dict = self._graph_data['graph_dict']
        entities_data = graph_dict['entities']
        relations = graph_dict['relations']
        
        if len(entities) < 2:
            return {
                'query': f"Commonality analysis for {entities}",
                'error': 'Need at least 2 entities to compare',
                'entities_found': entities
            }
        
        # Find entities in graph (case-insensitive)
        found_entities = {}
        for entity in entities:
            entity_lower = entity.lower()
            for entity_id, entity_data in entities_data.items():
                if entity_lower in entity_id.lower() or entity_id.lower().endswith(entity_lower):
                    found_entities[entity] = {
                        'id': entity_id,
                        'data': entity_data
                    }
                    break
        
        if len(found_entities) < 2:
            return {
                'query': f"Commonality analysis for {entities}",
                'error': f'Only found {len(found_entities)} entities in graph: {list(found_entities.keys())}',
                'entities_requested': entities,
                'entities_found': list(found_entities.keys())
            }
        
        # Analyze commonalities
        commonalities = self._analyze_entity_commonalities(found_entities, relations)
        
        return {
            'query': f"Commonalities between {list(found_entities.keys())}",
            'entities_analyzed': list(found_entities.keys()),
            'commonalities': commonalities,
            'summary': f"Found {len(commonalities)} types of commonalities"
        }
    
    def _analyze_entity_commonalities(self, entities: Dict[str, Dict], relations: List[Dict]) -> Dict[str, Any]:
        """Analyze commonalities between entities."""
        commonalities = {}
        
        # 1. Package commonalities
        packages = {}
        for name, entity in entities.items():
            package = entity['data']['properties'].get('package_name', 'unknown')
            if package not in packages:
                packages[package] = []
            packages[package].append(name)
        
        commonalities['packages'] = {
            'shared_packages': [pkg for pkg, models in packages.items() if len(models) > 1],
            'all_packages': packages,
            'analysis': f"{len([pkg for pkg, models in packages.items() if len(models) > 1])} shared packages"
        }
        
        # 2. Resource type commonalities  
        resource_types = {}
        for name, entity in entities.items():
            resource_type = entity['data']['properties'].get('resource_type', 'unknown')
            if resource_type not in resource_types:
                resource_types[resource_type] = []
            resource_types[resource_type].append(name)
        
        commonalities['resource_types'] = {
            'shared_types': [rtype for rtype, models in resource_types.items() if len(models) > 1],
            'all_types': resource_types,
            'analysis': f"{len([rtype for rtype, models in resource_types.items() if len(models) > 1])} shared resource types"
        }
        
        # 3. Schema/Database commonalities
        schemas = {}
        databases = {}
        for name, entity in entities.items():
            schema = entity['data']['properties'].get('schema', 'unknown')
            database = entity['data']['properties'].get('database', 'unknown')
            
            if schema not in schemas:
                schemas[schema] = []
            schemas[schema].append(name)
            
            if database not in databases:
                databases[database] = []
            databases[database].append(name)
        
        commonalities['schemas'] = {
            'shared_schemas': [schema for schema, models in schemas.items() if len(models) > 1],
            'all_schemas': schemas,
            'analysis': f"{len([schema for schema, models in schemas.items() if len(models) > 1])} shared schemas"
        }
        
        commonalities['databases'] = {
            'shared_databases': [db for db, models in databases.items() if len(models) > 1],
            'all_databases': databases,
            'analysis': f"{len([db for db, models in databases.items() if len(models) > 1])} shared databases"
        }
        
        # 4. Dependency commonalities (shared upstream dependencies)
        entity_ids = [entity['id'] for entity in entities.values()]
        shared_upstream = self._find_shared_dependencies(entity_ids, relations, 'upstream')
        shared_downstream = self._find_shared_dependencies(entity_ids, relations, 'downstream')
        
        commonalities['dependencies'] = {
            'shared_upstream': shared_upstream,
            'shared_downstream': shared_downstream,
            'analysis': f"{len(shared_upstream)} shared upstream, {len(shared_downstream)} shared downstream"
        }
        
        # 5. Tag commonalities
        all_tags = {}
        for name, entity in entities.items():
            tags = entity['data']['properties'].get('tags', [])
            for tag in tags:
                if tag not in all_tags:
                    all_tags[tag] = []
                all_tags[tag].append(name)
        
        shared_tags = [tag for tag, models in all_tags.items() if len(models) > 1]
        commonalities['tags'] = {
            'shared_tags': shared_tags,
            'all_tags': all_tags,
            'analysis': f"{len(shared_tags)} shared tags"
        }
        
        return commonalities
    
    def _find_shared_dependencies(self, entity_ids: List[str], relations: List[Dict], direction: str) -> List[Dict]:
        """Find shared upstream or downstream dependencies."""
        dependencies_by_entity = {}
        
        for entity_id in entity_ids:
            dependencies_by_entity[entity_id] = set()
            
            for rel in relations:
                source = rel.get('source', '')
                target = rel.get('target', '')
                label = rel.get('label', 'unknown')
                
                if direction == 'upstream':
                    # Find what this entity depends on
                    if label == 'depends_on' and entity_id in source:
                        dependencies_by_entity[entity_id].add(target)
                    elif label == 'flows_to' and entity_id in target:
                        dependencies_by_entity[entity_id].add(source)
                else:  # downstream
                    # Find what depends on this entity
                    if label == 'depends_on' and entity_id in target:
                        dependencies_by_entity[entity_id].add(source)
                    elif label == 'flows_to' and entity_id in source:
                        dependencies_by_entity[entity_id].add(target)
        
        # Find intersection of all dependency sets
        if not dependencies_by_entity:
            return []
        
        shared_deps = set.intersection(*dependencies_by_entity.values()) if dependencies_by_entity else set()
        
        return [{'dependency': dep, 'shared_by': entity_ids} for dep in shared_deps]

def main():
    """Command line interface for graph queries."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python graph_query_helper.py deps <model_name>     # Find dependencies")
        print("  python graph_query_helper.py upstream <model_name> # Find upstream deps")
        print("  python graph_query_helper.py search <term>         # Search models")
        print("  python graph_query_helper.py common <model1> <model2> [model3...] # Find commonalities")
        print()
        print("Examples:")
        print("  python graph_query_helper.py deps customer_phases_shared")
        print("  python graph_query_helper.py search customer")
        print("  python graph_query_helper.py common customer_phases_shared customer_segments_base")
        return
    
    command = sys.argv[1]
    
    helper = GraphQueryHelper()
    
    if command == "deps":
        if len(sys.argv) < 3:
            print("Usage: python graph_query_helper.py deps <model_name>")
            return
        term = sys.argv[2]
        result = helper.find_dependencies(term)
        print(f"\n🎯 {result['query']}")
        print("=" * 50)
        print(f"📊 {result['summary']}")
        
        if result['direct_dependencies']:
            print(f"\n✅ Direct Dependencies ({len(result['direct_dependencies'])}):")
            for dep in result['direct_dependencies']:
                print(f"  • {dep['dependent_model']}")
                print(f"    → {dep['relationship_type']}: {dep['depends_on']}")
        
        if result['external_dependencies']:
            print(f"\n🔗 External Dependencies ({len(result['external_dependencies'])}):")
            for dep in result['external_dependencies'][:10]:  # Show first 10
                print(f"  • {dep['dependent_model']}")
                print(f"    → {dep['relationship_type']}: {dep['depends_on']}")
            
            if len(result['external_dependencies']) > 10:
                print(f"    ... and {len(result['external_dependencies']) - 10} more")
    
    elif command == "upstream":
        if len(sys.argv) < 3:
            print("Usage: python graph_query_helper.py upstream <model_name>")
            return
        term = sys.argv[2]
        result = helper.find_upstream(term)
        print(f"\n🎯 {result['query']}")
        print("=" * 50)
        
        if result['upstream_dependencies']:
            print(f"📈 Upstream Dependencies ({result['count']}):")
            for dep in result['upstream_dependencies']:
                print(f"  • {dep['model']}")
                print(f"    → depends on: {dep['depends_on']} ({dep['relationship_type']})")
        else:
            print("No upstream dependencies found.")
    
    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python graph_query_helper.py search <term>")
            return
        term = sys.argv[2]
        models = helper.search_models(term)
        print(f"\n🔍 Models containing '{term}' ({len(models)}):")
        print("=" * 50)
        for model in models[:20]:  # Show first 20
            print(f"  • {model}")
        
        if len(models) > 20:
            print(f"    ... and {len(models) - 20} more")
    
    elif command == "common":
        if len(sys.argv) < 4:
            print("Usage: python graph_query_helper.py common <entity1> <entity2> [entity3...]")
            return
        entities = sys.argv[2:]
        result = helper.find_commonalities(entities)
        
        print(f"\n🔗 {result['query']}")
        print("=" * 60)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"📊 {result['summary']}")
        print(f"🎯 Entities analyzed: {', '.join(result['entities_analyzed'])}")
        
        commonalities = result['commonalities']
        
        # Package commonalities
        if commonalities['packages']['shared_packages']:
            print(f"\n📦 Shared Packages:")
            for pkg in commonalities['packages']['shared_packages']:
                models = commonalities['packages']['all_packages'][pkg]
                print(f"  • {pkg}: {', '.join(models)}")
        
        # Resource type commonalities
        if commonalities['resource_types']['shared_types']:
            print(f"\n🏷️ Shared Resource Types:")
            for rtype in commonalities['resource_types']['shared_types']:
                entities = commonalities['resource_types']['all_types'][rtype]
                print(f"  • {rtype}: {', '.join(entities)}")
        
        # Schema/Database commonalities
        if commonalities['schemas']['shared_schemas']:
            print(f"\n🗂️ Shared Schemas:")
            for schema in commonalities['schemas']['shared_schemas']:
                entities = commonalities['schemas']['all_schemas'][schema]
                print(f"  • {schema}: {', '.join(entities)}")
        
        if commonalities['databases']['shared_databases']:
            print(f"\n🗄️ Shared Databases:")
            for db in commonalities['databases']['shared_databases']:
                entities = commonalities['databases']['all_databases'][db]
                print(f"  • {db}: {', '.join(entities)}")
        
        # Dependency commonalities
        deps = commonalities['dependencies']
        if deps['shared_upstream']:
            print(f"\n⬆️ Shared Upstream Dependencies:")
            for dep in deps['shared_upstream']:
                print(f"  • {dep['dependency']}")
        
        if deps['shared_downstream']:
            print(f"\n⬇️ Shared Downstream Dependencies:")
            for dep in deps['shared_downstream']:
                print(f"  • {dep['dependency']}")
        
        # Tag commonalities
        if commonalities['tags']['shared_tags']:
            print(f"\n🏷️ Shared Tags:")
            for tag in commonalities['tags']['shared_tags']:
                entities = commonalities['tags']['all_tags'][tag]
                print(f"  • {tag}: {', '.join(entities)}")

if __name__ == "__main__":
    main()
