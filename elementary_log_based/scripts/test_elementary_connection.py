#!/usr/bin/env python3
"""
Test script to connect to Elementary tables in eio_ingest.elementary_logs
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_extractor import ElementaryDataExtractor


def main():
    """Test connection to Elementary tables."""
    print("🧪 Testing Elementary Connection")
    print("=" * 40)
    
    try:
        # Use the profile that connects to eio_ingest database
        print("🔍 Looking for profiles that connect to eio_ingest database...")
        
        # Based on the profile checker output, these profiles connect to eio_ingest:
        test_configs = [
            ('test_project', 'test_project'),  # eio_ingest.user_marquein_transform
            ('default', 'stg_perso'),  # eio_ingest.user_marquein_transform  
            ('default', 'prod')  # EIO_INGEST.ENGAGEMENT_TRANSFORM
        ]
        
        for profile_name, target in test_configs:
            print(f"\n📋 Testing {profile_name}.{target}...")
            
            try:
                # Initialize extractor
                extractor = ElementaryDataExtractor(profile_name=profile_name, target=target)
                
                # Manually set the elementary schema to 'elementary_log'
                extractor.set_elementary_schema('elementary_log')
                
                # Try to connect and extract a small sample
                extractor.connect()
                
                # Test by trying to get a count of models
                print("   🔍 Testing elementary_log.dbt_models table...")
                query = "SELECT COUNT(*) as model_count FROM elementary_log.dbt_models LIMIT 1"
                
                cursor = extractor.connection.cursor()
                cursor.execute(query)
                result = cursor.fetchone()
                cursor.close()
                
                if result:
                    model_count = result[0]
                    print(f"   ✅ Found {model_count} models in elementary_log.dbt_models!")
                    
                    # Try extracting a small sample
                    print("   📊 Extracting sample data...")
                    models_df = extractor.extract_models()
                    
                    if not models_df.empty:
                        print(f"   🎉 Successfully extracted {len(models_df)} models!")
                        print(f"   📋 Sample columns: {list(models_df.columns)[:5]}")
                        
                        # Show a sample model
                        if len(models_df) > 0:
                            sample_model = models_df.iloc[0]
                            print(f"   📦 Sample model: {sample_model['name']}")
                        
                        print(f"\n✅ SUCCESS! Elementary tables found in {profile_name}.{target}")
                        print(f"   Database: {extractor.config.get('database')}")
                        print(f"   Schema: elementary_log")
                        
                        extractor.disconnect()
                        return profile_name, target
                    else:
                        print("   ⚠️ No models returned from query")
                        
                extractor.disconnect()
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                if 'does not exist' in str(e) or 'not found' in str(e):
                    print("   💡 elementary_log schema not found in this database")
                continue
        
        print("\n❌ No Elementary tables found in any of the tested profiles.")
        print("💡 Make sure Elementary is installed and the tables are in 'elementary_log' schema")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()