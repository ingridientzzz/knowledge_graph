"""
Database configuration for connecting to Snowflake Elementary tables.
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class DatabaseConfig:
    """Configuration class for database connections."""
    
    @staticmethod
    def get_snowflake_config(profile_name: str = None, target: str = 'dev') -> Dict[str, Any]:
        """Get Snowflake connection configuration from dbt profiles.yml."""
        profiles_path = Path.home() / '.dbt' / 'profiles.yml'
        
        if not profiles_path.exists():
            raise FileNotFoundError(f"dbt profiles.yml not found at {profiles_path}")
        
        try:
            with open(profiles_path, 'r') as f:
                profiles = yaml.safe_load(f)
            
            # If profile_name not specified, try to find a Snowflake profile
            if profile_name is None:
                profile_name = DatabaseConfig._find_snowflake_profile(profiles)
                if profile_name is None:
                    raise ValueError("No Snowflake profile found in profiles.yml")
            
            if profile_name not in profiles:
                raise ValueError(f"Profile '{profile_name}' not found in profiles.yml")
            
            profile = profiles[profile_name]
            
            # Get the target configuration
            if 'outputs' not in profile:
                raise ValueError(f"No outputs found in profile '{profile_name}'")
            
            if target not in profile['outputs']:
                available_targets = list(profile['outputs'].keys())
                raise ValueError(f"Target '{target}' not found. Available targets: {available_targets}")
            
            target_config = profile['outputs'][target]
            
            # Validate it's a Snowflake connection
            if target_config.get('type') != 'snowflake':
                raise ValueError(f"Target '{target}' is not a Snowflake connection (type: {target_config.get('type')})")
            
            # Build Snowflake connection config
            snowflake_config = {
                'user': target_config.get('user'),
                'password': target_config.get('password'),
                'account': target_config.get('account'),
                'warehouse': target_config.get('warehouse'),
                'database': target_config.get('database'),
                'schema': target_config.get('schema'),
                'role': target_config.get('role'),
            }
            
            # Handle authenticator if present (for SSO, etc.)
            if 'authenticator' in target_config:
                snowflake_config['authenticator'] = target_config['authenticator']
            
            # Handle private key authentication
            if 'private_key_path' in target_config:
                snowflake_config['private_key_path'] = target_config['private_key_path']
            if 'private_key_passphrase' in target_config:
                snowflake_config['private_key_passphrase'] = target_config['private_key_passphrase']
            
            # Remove None values
            snowflake_config = {k: v for k, v in snowflake_config.items() if v is not None}
            
            print(f"✅ Loaded Snowflake config from dbt profile '{profile_name}' target '{target}'")
            print(f"   Database: {snowflake_config.get('database')}")
            print(f"   Schema: {snowflake_config.get('schema')}")
            print(f"   Warehouse: {snowflake_config.get('warehouse')}")
            
            return snowflake_config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing profiles.yml: {e}")
        except Exception as e:
            raise ValueError(f"Error reading dbt profiles: {e}")
    
    @staticmethod
    def _find_snowflake_profile(profiles: Dict[str, Any]) -> Optional[str]:
        """Find the first Snowflake profile in the profiles.yml."""
        for profile_name, profile_config in profiles.items():
            if isinstance(profile_config, dict) and 'outputs' in profile_config:
                for target_name, target_config in profile_config['outputs'].items():
                    if isinstance(target_config, dict) and target_config.get('type') == 'snowflake':
                        return profile_name
        return None
    
    @staticmethod
    def list_available_profiles() -> Dict[str, Any]:
        """List all available dbt profiles and their targets."""
        profiles_path = Path.home() / '.dbt' / 'profiles.yml'
        
        if not profiles_path.exists():
            return {}
        
        try:
            with open(profiles_path, 'r') as f:
                profiles = yaml.safe_load(f)
            
            result = {}
            for profile_name, profile_config in profiles.items():
                if isinstance(profile_config, dict) and 'outputs' in profile_config:
                    targets = {}
                    for target_name, target_config in profile_config['outputs'].items():
                        if isinstance(target_config, dict):
                            targets[target_name] = {
                                'type': target_config.get('type'),
                                'database': target_config.get('database'),
                                'schema': target_config.get('schema')
                            }
                    result[profile_name] = {
                        'default_target': profile_config.get('target'),
                        'targets': targets
                    }
            
            return result
            
        except Exception as e:
            print(f"Error reading profiles: {e}")
            return {}
    
    @staticmethod
    def get_neo4j_config() -> Dict[str, Any]:
        """Get Neo4j connection configuration."""
        return {
            'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            'user': os.getenv('NEO4J_USER', 'neo4j'),
            'password': os.getenv('NEO4J_PASSWORD', 'password'),
        }