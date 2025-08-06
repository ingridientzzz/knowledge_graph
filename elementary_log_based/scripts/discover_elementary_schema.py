#!/usr/bin/env python3
"""
Discover the actual schema structure of Elementary tables.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_extractor import ElementaryDataExtractor


def main():
    """Discover Elementary table schemas."""
    print("🔍 Discovering Elementary Table Schema")
    print("=" * 40)
    
    try:
        # Use a profile that we know connects to eio_ingest
        print("📋 Using test_project.test_project profile...")
        
        extractor = ElementaryDataExtractor(profile_name='test_project', target='test_project')
        extractor.set_elementary_schema('elementary_log')
        extractor.connect()
        
        # List of Elementary tables to check
        tables_to_check = [
            'dbt_models',
            'dbt_tests', 
            'elementary_test_results',
            'dbt_run_results',
            'dbt_invocations',
            'dbt_sources',
            'model_columns'
        ]
        
        cursor = extractor.connection.cursor()
        
        for table_name in tables_to_check:
            print(f"\n📊 Checking table: elementary_log.{table_name}")
            
            try:
                # Get column information
                describe_query = f"DESCRIBE TABLE elementary_log.{table_name}"
                cursor.execute(describe_query)
                columns = cursor.fetchall()
                
                if columns:
                    print(f"   ✅ Found {len(columns)} columns:")
                    for i, col in enumerate(columns[:10]):  # Show first 10 columns
                        col_name = col[0]
                        col_type = col[1] if len(col) > 1 else 'unknown'
                        print(f"      {i+1:2d}. {col_name} ({col_type})")
                    
                    if len(columns) > 10:
                        print(f"      ... and {len(columns) - 10} more columns")
                    
                    # Get a sample row count
                    count_query = f"SELECT COUNT(*) FROM elementary_log.{table_name}"
                    cursor.execute(count_query)
                    count = cursor.fetchone()[0]
                    print(f"   📈 Row count: {count}")
                    
                    # Show a sample of the first few columns
                    if count > 0:
                        sample_cols = [col[0] for col in columns[:5]]
                        sample_query = f"SELECT {', '.join(sample_cols)} FROM elementary_log.{table_name} LIMIT 3"
                        try:
                            cursor.execute(sample_query)
                            samples = cursor.fetchall()
                            print(f"   📋 Sample data (first {len(sample_cols)} columns):")
                            for j, sample in enumerate(samples):
                                sample_str = ', '.join([str(s)[:50] + ('...' if len(str(s)) > 50 else '') for s in sample])
                                print(f"      Row {j+1}: {sample_str}")
                        except Exception as sample_error:
                            print(f"   ⚠️ Could not fetch sample data: {sample_error}")
                
                else:
                    print("   ⚠️ No columns found")
                    
            except Exception as e:
                print(f"   ❌ Error checking {table_name}: {e}")
                continue
        
        cursor.close()
        extractor.disconnect()
        
        print(f"\n🎉 Schema discovery completed!")
        print(f"💡 Now we can update the data extraction queries to match your schema")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()