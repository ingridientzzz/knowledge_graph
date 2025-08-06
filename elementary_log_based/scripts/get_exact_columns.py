#!/usr/bin/env python3
"""
Get exact column names for specific Elementary tables.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_extractor import ElementaryDataExtractor


def main():
    """Get exact column names."""
    print("🔍 Getting Exact Column Names")
    print("=" * 40)
    
    try:
        extractor = ElementaryDataExtractor(profile_name='test_project', target='test_project')
        extractor.set_elementary_schema('elementary_log')
        extractor.connect()
        
        cursor = extractor.connection.cursor()
        
        # Check specific tables with detailed column info
        tables_to_fix = {
            'dbt_tests': ['test_type', 'test_params'],
            'elementary_test_results': ['execution_time', 'compiled_sql', 'test_message'],
            'dbt_invocations': ['is_full_refresh', 'env_vars']
        }
        
        for table_name, missing_cols in tables_to_fix.items():
            print(f"\n📊 Full columns for {table_name}:")
            cursor.execute(f"DESCRIBE TABLE elementary_log.{table_name}")
            columns = cursor.fetchall()
            
            all_columns = [col[0] for col in columns]
            print(f"   All columns: {all_columns}")
            
            print(f"   Looking for: {missing_cols}")
            for col in missing_cols:
                matches = [c for c in all_columns if col.lower() in c.lower()]
                if matches:
                    print(f"   ✅ '{col}' might be: {matches}")
                else:
                    print(f"   ❌ '{col}' not found")
        
        cursor.close()
        extractor.disconnect()
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()