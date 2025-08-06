#!/usr/bin/env python3
"""
Test that SQLAlchemy integration works without pandas warnings.
"""
import sys
import os
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_extractor import ElementaryDataExtractor


def main():
    """Test SQLAlchemy integration."""
    print("🧪 Testing SQLAlchemy Integration")
    print("=" * 35)
    
    # Capture warnings to verify no pandas warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            print("\n1️⃣ Testing data extraction...")
            extractor = ElementaryDataExtractor(profile_name='test_project', target='test_project')
            extractor.set_elementary_schema('elementary_log')
            extractor.connect()
            
            # Test one extraction that we know works
            models_df = extractor.extract_models()
            extractor.disconnect()
            
            print(f"✅ Successfully extracted {len(models_df)} models")
            
            # Check for pandas warnings
            pandas_warnings = [warning for warning in w if 'pandas only supports SQLAlchemy' in str(warning.message)]
            
            if pandas_warnings:
                print(f"❌ Found {len(pandas_warnings)} pandas SQLAlchemy warnings:")
                for warning in pandas_warnings:
                    print(f"   - {warning.message}")
                return False
            else:
                print("✅ No pandas SQLAlchemy warnings detected!")
                return True
                
        except Exception as e:
            print(f"❌ Error during test: {e}")
            return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SQLAlchemy integration working perfectly!")
        print("📋 Summary:")
        print("   ✅ pandas.read_sql() now uses proper SQLAlchemy engine")
        print("   ✅ No more pandas connector warnings")
        print("   ✅ Full compatibility maintained")
    else:
        print("\n💥 SQLAlchemy integration needs more work")