#!/usr/bin/env python3
"""
Utility script to check and list available dbt profiles.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.database_config import DatabaseConfig
import json


def main():
    """Check dbt profiles and test connection."""
    print("🔍 dbt Profiles Checker")
    print("=" * 40)
    
    try:
        # List all available profiles
        profiles = DatabaseConfig.list_available_profiles()
        
        if not profiles:
            print("❌ No dbt profiles found at ~/.dbt/profiles.yml")
            print("\n💡 Make sure you have:")
            print("   1. dbt installed")
            print("   2. A profiles.yml file at ~/.dbt/profiles.yml")
            print("   3. At least one Snowflake connection configured")
            return
        
        print(f"✅ Found {len(profiles)} dbt profile(s):")
        print()
        
        snowflake_profiles = []
        
        for profile_name, profile_info in profiles.items():
            print(f"📋 Profile: {profile_name}")
            print(f"   Default target: {profile_info.get('default_target', 'unknown')}")
            print(f"   Available targets:")
            
            for target_name, target_info in profile_info['targets'].items():
                connection_type = target_info.get('type', 'unknown')
                database = target_info.get('database', 'unknown')
                schema = target_info.get('schema', 'unknown')
                
                status_icon = "❄️" if connection_type == 'snowflake' else "🔧"
                print(f"      {status_icon} {target_name} ({connection_type}): {database}.{schema}")
                
                if connection_type == 'snowflake':
                    snowflake_profiles.append((profile_name, target_name))
            print()
        
        if not snowflake_profiles:
            print("⚠️  No Snowflake profiles found!")
            print("   The knowledge graph requires a Snowflake connection with Elementary tables.")
            return
        
        # Test Snowflake connections
        print("🧪 Testing Snowflake connections...")
        print()
        
        for profile_name, target_name in snowflake_profiles:
            try:
                print(f"Testing {profile_name}.{target_name}...")
                config = DatabaseConfig.get_snowflake_config(profile_name, target_name)
                
                # Test basic connection info
                print(f"   ✅ Configuration loaded successfully")
                print(f"      Account: {config.get('account', 'N/A')}")
                print(f"      Database: {config.get('database', 'N/A')}")
                print(f"      Schema: {config.get('schema', 'N/A')}")
                print(f"      Warehouse: {config.get('warehouse', 'N/A')}")
                
                # Try to import and test connection (optional)
                try:
                    import snowflake.connector
                    conn = snowflake.connector.connect(**config)
                    cursor = conn.cursor()
                    cursor.execute("SELECT CURRENT_VERSION()")
                    version = cursor.fetchone()[0]
                    cursor.close()
                    conn.close()
                    print(f"   🎉 Connection successful! Snowflake version: {version}")
                    
                    # Check for Elementary tables
                    print(f"   🔍 Checking for Elementary tables...")
                    conn = snowflake.connector.connect(**config)
                    cursor = conn.cursor()
                    
                    # Common elementary schema patterns
                    base_schema = config.get('schema', '')
                    possible_schemas = [
                        f"{base_schema}_elementary",
                        "elementary",
                        f"{base_schema.upper()}_ELEMENTARY",
                        "ELEMENTARY"
                    ]
                    
                    found_elementary = False
                    for schema in possible_schemas:
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {schema}.dbt_models LIMIT 1")
                            count = cursor.fetchone()[0]
                            print(f"   📊 Found Elementary tables in schema: {schema} ({count} models)")
                            found_elementary = True
                            break
                        except:
                            continue
                    
                    if not found_elementary:
                        print(f"   ⚠️  No Elementary tables found. Make sure Elementary is installed and run.")
                    
                    cursor.close()
                    conn.close()
                    
                except ImportError:
                    print(f"   ⚠️  snowflake-connector-python not installed, skipping connection test")
                except Exception as conn_error:
                    print(f"   ❌ Connection failed: {conn_error}")
                
                print()
                
            except Exception as e:
                print(f"   ❌ Failed to load configuration: {e}")
                print()
        
        # Recommendations
        print("💡 Recommendations:")
        print("   1. Use a Snowflake profile with Elementary tables")
        print("   2. Make sure the Elementary dbt package is installed in your project")
        print("   3. Run `dbt run --select elementary` to create the tables")
        print("   4. Use the profile name and target in your knowledge graph scripts")
        
        print("\n🚀 Ready to use with ElementaryDataExtractor!")
        if snowflake_profiles:
            recommended = snowflake_profiles[0]
            print(f"   Example: ElementaryDataExtractor(profile_name='{recommended[0]}', target='{recommended[1]}')")
        
    except Exception as e:
        print(f"❌ Error checking profiles: {e}")


if __name__ == "__main__":
    main()