"""
Intelligent Query Router for ChatDBT
Routes queries to optimal search strategies based on user intent
"""

import re
from typing import Dict, Tuple, List
from enum import Enum

class QueryIntent(Enum):
    """Types of query intents for routing"""
    PACKAGE_LIST = "package_list"          # "list packages", "show packages"
    PACKAGE_DETAIL = "package_detail"      # "tell me about package X"
    MODEL_SEARCH = "model_search"          # "show models", "find model X"
    SOURCE_SEARCH = "source_search"        # "show sources", "what sources"
    TEST_SEARCH = "test_search"            # "show tests", "what tests"
    LINEAGE = "lineage"                    # "dependencies", "lineage", "downstream"
    GRAPH_DEPENDENCY = "graph_dependency"  # "what depends on X", "dependencies of Y"
    GRAPH_SEARCH = "graph_search"          # "find models with X", "search for Y"
    GRAPH_COMMONALITY = "graph_commonality" # "what do X and Y have in common", "commonalities between"
    SQL_CODE = "sql_code"                  # "show SQL", "how is X calculated"
    GREETING = "greeting"                  # "hello", "hi", "thanks"
    GENERAL = "general"                    # Complex or unclear intent

class QueryMethod(Enum):
    """Methods for handling queries"""
    VECTOR_SEARCH = "vector_search"        # Use vector store + LLM
    GRAPH_QUERY = "graph_query"            # Use direct graph store query
    HYBRID = "hybrid"                      # Use both methods

class QueryRouter:
    """Routes queries to optimal retrieval strategies"""
    
    def __init__(self):
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.PACKAGE_LIST: [
                r'\b(list|show|all|what)\s+(packages?|pkgs?)\b',
                r'\bpackages?\s+(available|exist)\b',
                r'\bhow many packages?\b'
            ],
            QueryIntent.PACKAGE_DETAIL: [
                r'\b(package|pkg)\s+(\w+)',
                r'\b(tell me about|describe|explain)\s+(\w+)\s+(package|pkg)\b',
                r'\bwhat.{0,20}(\w+)\s+package\b'
            ],
            QueryIntent.MODEL_SEARCH: [
                r'\b(list|show|all|what)\s+(models?|tables?)\b',
                r'\bmodels?\s+(in|from|available)\b',
                r'\bfind.{0,20}model\b',
                r'\b(models?|tables?)\s+in\s+(the\s+)?(\w+)\s+(package|pkg)\b',
                r'\b(all|show|list)\s+(models?|tables?)\s+(in|from)\s+(\w+)\b'
            ],
            QueryIntent.SOURCE_SEARCH: [
                r'\b(list|show|all|what)\s+sources?\b',
                r'\bsources?\s+(available|exist)\b',
                r'\bexternal.{0,20}(data|tables?)\b'
            ],
            QueryIntent.TEST_SEARCH: [
                r'\b(list|show|all|what)\s+tests?\b',
                r'\btests?\s+(for|on|applied)\b',
                r'\bdata quality\b'
            ],
            QueryIntent.LINEAGE: [
                r'\b(lineage|flow|pipeline)\b',
                r'\b(impact|affected|changes?)\b',
                r'\bdata.{0,20}flow\b'
            ],
            QueryIntent.GRAPH_DEPENDENCY: [
                r'\bwhat.{0,30}(depends?\s+on|uses?|references?)\s+\w+',
                r'\bmodels?.{0,30}(depend|use|reference).{0,20}\w+',
                r'\bwho.{0,20}(uses?|depends?\s+on)\s+\w+',
                r'\b\w+.{0,20}(dependencies|dependents?)\b',
                r'\b(upstream|downstream).{0,20}(of|from|for)\s+\w+',
                r'\bdepend.{0,20}on\s+\w+',
                r'\bwhat\s+does\s+\w+\s+depend\s+on',
                r'\b\w+\s+(upstream|dependencies)',
                r'\bshow.{0,20}(upstream|dependencies).{0,20}(of|for)\s+\w+'
            ],
            QueryIntent.GRAPH_SEARCH: [
                r'\b(find|search|list).{0,20}models?.{0,20}(with|containing|like)\s+\w+',
                r'\bmodels?.{0,20}(containing|with|named|called)\s+\w+',
                r'\b(show|list).{0,20}(all|models?).{0,20}(with|containing)\s+\w+',
                r'\bsearch.{0,20}(for|models?)\s+\w+'
            ],
            QueryIntent.GRAPH_COMMONALITY: [
                r'\bwhat.{0,20}(do|are).{0,30}(have\s+in\s+common|common|shared)',
                r'\b(commonalities|similarities)\s+(between|of|among)',
                r'\b(compare|comparison)\s+(between|of)\s+\w+.{0,20}(and|vs)\s+\w+',
                r'\bshared.{0,20}(features|properties|attributes|dependencies)',
                r'\bin\s+common\s+(between|with)',
                r'\bhow.{0,20}(similar|alike)\s+(are|is)\s+\w+.{0,20}(and|to)\s+\w+',
                r'\bwhat.{0,20}\w+.{0,20}(and|&)\s+\w+.{0,20}(share|common)',
                r'\bboth\s+\w+.{0,20}(and|&)\s+\w+.{0,20}(have|use|share)'
            ],
            QueryIntent.SQL_CODE: [
                r'\b(SQL|code|query|how.{0,20}calculated)\b',
                r'\bshow.{0,20}(transformation|logic)\b',
                r'\bhow.{0,20}(built|created|computed)\b'
            ],
            QueryIntent.GREETING: [
                r'\b(hi|hello|hey|thanks?|bye)\b',
                r'^(good|morning|afternoon|evening)\b',
                r'\b(help|assistance)\b$'
            ]
        }
    
    def classify_query(self, query: str) -> QueryIntent:
        """Classify query intent based on patterns"""
        query_lower = query.lower().strip()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return intent
        
        return QueryIntent.GENERAL
    
    def get_optimal_method(self, query: str) -> QueryMethod:
        """Determine the best method for handling the query"""
        intent = self.classify_query(query)
        
        # Graph queries are best for exact relationship questions
        if intent in [QueryIntent.GRAPH_DEPENDENCY, QueryIntent.GRAPH_SEARCH, QueryIntent.GRAPH_COMMONALITY]:
            return QueryMethod.GRAPH_QUERY
        
        # Lineage might benefit from hybrid approach
        elif intent == QueryIntent.LINEAGE:
            return QueryMethod.HYBRID
        
        # Everything else uses vector search for natural language understanding
        else:
            return QueryMethod.VECTOR_SEARCH
    
    def extract_model_name(self, query: str) -> str:
        """Extract model name from dependency queries"""
        # Look for patterns like "depends on X", "uses Y", "what does X depend on", etc.
        patterns = [
            r'(?:depends?\s+on|uses?|references?)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:dependencies|dependents?)',
            r'(?:upstream|downstream)(?:\s+(?:of|from|for))?\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'(?:models?\s+(?:depend|use|reference))\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'(?:who\s+(?:uses?|depends?\s+on))\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'what\s+does\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+depend\s+on',
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:upstream|dependencies)',
            r'(?:upstream|dependencies)(?:\s+(?:of|for))\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        query_lower = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)
        
        # Fallback: look for any word that might be a model name
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query)
        # Filter out common words
        common_words = {'what', 'models', 'depend', 'on', 'uses', 'the', 'show', 'me', 'find', 'list', 'does'}
        candidates = [w for w in words if w.lower() not in common_words and len(w) > 2]
        
        return candidates[0] if candidates else ""

    def extract_entities_for_commonality(self, query: str) -> List[str]:
        """Extract multiple entity names from commonality queries."""
        # Look for patterns with multiple entities connected by 'and', 'vs', '&', etc.
        patterns = [
            r'(between|of|among)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(and|&|vs)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s+(and|&|vs)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(have|share|common)',
            r'(compare|comparison)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(and|to|vs|with)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'both\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(and|&)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'what\s+do\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(and|&)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        query_lower = query.lower()
        entities = []
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                # Extract entities from different capture groups depending on pattern
                groups = match.groups()
                for group in groups:
                    if group and group not in ['between', 'of', 'among', 'and', '&', 'vs', 'with', 'to', 'have', 'share', 'common', 'compare', 'comparison', 'both', 'what', 'do']:
                        if len(group) > 2 and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', group):
                            entities.append(group)
                break
        
        # If no pattern match, try to find multiple model-like words
        if not entities:
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b', query)
            common_words = {'what', 'models', 'depend', 'common', 'shared', 'between', 'have', 'show', 'find', 'list', 'compare', 'similar', 'alike', 'both'}
            candidates = [w for w in words if w.lower() not in common_words]
            entities = candidates[:4]  # Limit to first 4 entities
        
        return list(dict.fromkeys(entities))  # Remove duplicates while preserving order

    def is_upstream_query(self, query: str) -> bool:
        """Determine if this is asking for upstream dependencies (what X depends on)"""
        upstream_patterns = [
            r'what\s+does\s+\w+\s+depend\s+on',
            r'\w+\s+(?:upstream|dependencies)(?:\s|$)',
            r'(?:upstream|dependencies)(?:\s+(?:of|for))\s+\w+',
            r'what\s+\w+\s+(?:uses?|references?|depends?\s+on)'
        ]
        
        query_lower = query.lower()
        for pattern in upstream_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def get_retrieval_config(self, query: str) -> Dict:
        """Get optimal retrieval configuration for query"""
        intent = self.classify_query(query)
        method = self.get_optimal_method(query)
        
        config = {
            'similarity_top_k': 10,      # Default
            'chunk_size': 4096,          # Default  
            'search_filter': None,       # No filtering
            'use_reranking': False,      # Advanced reranking
            'context_strategy': 'full'   # Full context
        }
        
        # Optimize based on intent
        if intent == QueryIntent.PACKAGE_LIST:
            config.update({
                'similarity_top_k': 25,          # Need more docs to get all package batches (38 packages / 8 per batch = 5 batches + master + buffer)
                'search_filter': 'packages_summary',  # Focus on package summaries
                'context_strategy': 'summary'
            })
            
        elif intent == QueryIntent.PACKAGE_DETAIL:
            config.update({
                'similarity_top_k': 8,
                'search_filter': 'package_specific',
                'context_strategy': 'detailed'
            })
            
        elif intent == QueryIntent.MODEL_SEARCH:
            config.update({
                'similarity_top_k': 15,          # More models to show
                'search_filter': 'models_only',
                'context_strategy': 'structured'
            })
            
        elif intent == QueryIntent.SOURCE_SEARCH:
            config.update({
                'similarity_top_k': 10,
                'search_filter': 'sources_only',
                'context_strategy': 'structured'
            })
            
        elif intent == QueryIntent.TEST_SEARCH:
            config.update({
                'similarity_top_k': 12,
                'search_filter': 'tests_only',
                'context_strategy': 'structured'
            })
            
        elif intent == QueryIntent.LINEAGE:
            config.update({
                'similarity_top_k': 20,          # Need more context for relationships
                'use_reranking': True,
                'context_strategy': 'relationship_focused'
            })
            
        elif intent == QueryIntent.GRAPH_DEPENDENCY:
            config.update({
                'similarity_top_k': 5,           # Graph query will provide exact results
                'context_strategy': 'graph_focused',
                'model_name': self.extract_model_name(query),
                'is_upstream_query': self.is_upstream_query(query)
            })
            
        elif intent == QueryIntent.GRAPH_SEARCH:
            config.update({
                'similarity_top_k': 5,           # Graph query will provide exact results
                'context_strategy': 'search_focused',
                'search_term': self.extract_model_name(query)
            })
            
        elif intent == QueryIntent.GRAPH_COMMONALITY:
            config.update({
                'similarity_top_k': 5,           # Graph query will provide exact results
                'context_strategy': 'commonality_focused',
                'entities': self.extract_entities_for_commonality(query)
            })
            
        elif intent == QueryIntent.SQL_CODE:
            config.update({
                'similarity_top_k': 8,
                'context_strategy': 'code_focused'
            })
            
        elif intent == QueryIntent.GREETING:
            config.update({
                'similarity_top_k': 2,           # Minimal context
                'context_strategy': 'minimal'
            })
        
        return {
            'intent': intent.value,
            'method': method.value,
            'config': config,
            'explanation': self._get_strategy_explanation(intent),
            'method_explanation': self._get_method_explanation(method)
        }
    
    def _get_strategy_explanation(self, intent: QueryIntent) -> str:
        """Get human-readable explanation of chosen strategy"""
        explanations = {
            QueryIntent.PACKAGE_LIST: "Package overview search - comprehensive package listing (top-k=25)",
            QueryIntent.PACKAGE_DETAIL: "Package detail search - comprehensive package information", 
            QueryIntent.MODEL_SEARCH: "Model discovery search - structured model listings",
            QueryIntent.SOURCE_SEARCH: "Source discovery search - external data sources",
            QueryIntent.TEST_SEARCH: "Test discovery search - data quality information",
            QueryIntent.LINEAGE: "Lineage analysis search - relationship mapping",
            QueryIntent.GRAPH_DEPENDENCY: "Graph dependency query - direct relationship lookup",
            QueryIntent.GRAPH_SEARCH: "Graph search query - direct model search",
            QueryIntent.GRAPH_COMMONALITY: "Graph commonality analysis - finding shared attributes/dependencies",
            QueryIntent.SQL_CODE: "Code analysis search - transformation logic",
            QueryIntent.GREETING: "Simple greeting - minimal context needed",
            QueryIntent.GENERAL: "General knowledge search - full context retrieval"
        }
        return explanations.get(intent, "Standard search strategy")
    
    def _get_method_explanation(self, method: QueryMethod) -> str:
        """Get human-readable explanation of chosen method"""
        explanations = {
            QueryMethod.VECTOR_SEARCH: "Using vector similarity search with LLM for natural language understanding",
            QueryMethod.GRAPH_QUERY: "Using direct graph store query for exact relationship data (fastest & most accurate)",
            QueryMethod.HYBRID: "Using both graph query and vector search for comprehensive results"
        }
        return explanations.get(method, "Standard vector search method")

# Global router instance
query_router = QueryRouter()

def route_query(query: str) -> Dict:
    """Main function to route queries with optimal configuration"""
    return query_router.get_retrieval_config(query)

# Test function
if __name__ == "__main__":
    test_queries = [
        "List all packages available",
        "Show me models in the rnk package", 
        "What are the dependencies for customer_metrics?",
        "Hello, how can you help me?",
        "Show me the SQL code for revenue calculation"
    ]
    
    print("🧪 Testing Query Router:")
    for query in test_queries:
        result = route_query(query)
        print(f"\nQuery: '{query}'")
        print(f"Intent: {result['intent']}")
        print(f"Top-K: {result['config']['similarity_top_k']}")
        print(f"Strategy: {result['explanation']}")
