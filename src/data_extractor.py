"""
Data extraction module for dbt Elementary tables from Snowflake.
"""
import pandas as pd
import snowflake.connector
from sqlalchemy import create_engine
from typing import Dict, List, Any, Optional
import json
import sys
import os
import urllib.parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database_config import DatabaseConfig


class ElementaryDataExtractor:
    """Extract data from dbt Elementary tables in Snowflake."""
    
    def __init__(self, profile_name: Optional[str] = None, target: str = 'dev', 
                 connection_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data extractor.
        
        Args:
            profile_name: dbt profile name (auto-detected if None)
            target: dbt target name (default: 'dev')
            connection_config: Manual connection config (overrides dbt profiles)
        """
        if connection_config:
            self.config = connection_config
        else:
            self.config = DatabaseConfig.get_snowflake_config(profile_name, target)
        
        self.connection = None
        self.engine = None
        self.elementary_schema = None
    
    def connect(self) -> None:
        """Establish connection to Snowflake."""
        try:
            # Create both direct connection (for metadata queries) and SQLAlchemy engine (for pandas)
            self.connection = snowflake.connector.connect(**self.config)
            
            # Create SQLAlchemy engine for pandas compatibility
            self.engine = self._create_sqlalchemy_engine()
            
            print("✅ Connected to Snowflake successfully")
            
            # Auto-detect elementary schema if not explicitly set
            if not self.elementary_schema:
                self.elementary_schema = self._detect_elementary_schema()
                
        except Exception as e:
            print(f"❌ Failed to connect to Snowflake: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close the Snowflake connection."""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        print("🔌 Disconnected from Snowflake")
    
    def _create_sqlalchemy_engine(self):
        """Create SQLAlchemy engine for pandas compatibility."""
        # Build connection string for Snowflake SQLAlchemy
        user = urllib.parse.quote_plus(self.config['user'])
        password = urllib.parse.quote_plus(self.config.get('password', ''))
        account = self.config['account']
        warehouse = self.config.get('warehouse', '')
        database = self.config.get('database', '')
        schema = self.config.get('schema', '')
        
        # Handle different authentication methods
        if 'authenticator' in self.config:
            # For SSO/external authentication, we don't need password in URL
            connection_string = f"snowflake://{user}@{account}/{database}/{schema}?warehouse={warehouse}&authenticator={self.config['authenticator']}"
        else:
            # For password authentication
            connection_string = f"snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"
        
        return create_engine(connection_string)
    
    def extract_models(self) -> pd.DataFrame:
        """Extract dbt models metadata."""
        query = f"""
        SELECT 
            unique_id,
            name,
            database_name,
            schema_name,
            alias,
            depends_on_nodes,
            package_name,
            materialization,
            generated_at,
            path,
            tags,
            meta,
            checksum,
            owner
        FROM {self.elementary_schema}.dbt_models
        WHERE generated_at >= CURRENT_DATE - 30  -- Last 30 days
        ORDER BY generated_at DESC
        """
        return self._execute_query(query, "models")
    
    def extract_tests(self) -> pd.DataFrame:
        """Extract dbt tests metadata."""
        query = f"""
        SELECT 
            unique_id,
            name,
            database_name,
            schema_name,
            depends_on_nodes,
            type as test_type,
            test_params,
            package_name,
            generated_at,
            path,
            tags,
            meta,
            alias,
            test_column_name,
            severity
        FROM {self.elementary_schema}.dbt_tests
        WHERE generated_at >= CURRENT_DATE - 30
        ORDER BY generated_at DESC
        """
        return self._execute_query(query, "tests")
    
    def extract_sources(self) -> pd.DataFrame:
        """Extract dbt sources metadata."""
        query = f"""
        SELECT 
            unique_id,
            name,
            source_name,
            database_name,
            schema_name,
            identifier,
            generated_at,
            tags,
            meta,
            loaded_at_field,
            freshness_warn_after,
            freshness_error_after
        FROM {self.elementary_schema}.dbt_sources
        WHERE generated_at >= CURRENT_DATE - 30
        ORDER BY generated_at DESC
        """
        return self._execute_query(query, "sources")
    
    def extract_test_results(self) -> pd.DataFrame:
        """Extract test execution results."""
        query = f"""
        SELECT 
            test_unique_id,
            model_unique_id,
            invocation_id,
            status,
            result_rows,
            detected_at,
            id,
            database_name,
            schema_name,
            test_type,
            test_results_description as test_message,
            test_name,
            severity,
            failures,
            failed_row_count
        FROM {self.elementary_schema}.elementary_test_results
        WHERE detected_at >= CURRENT_DATE - 30
        ORDER BY detected_at DESC
        """
        return self._execute_query(query, "test_results")
    
    def extract_run_results(self) -> pd.DataFrame:
        """Extract model execution results."""
        query = f"""
        SELECT 
            unique_id as model_unique_id,
            invocation_id,
            status,
            execution_time,
            rows_affected,
            materialization,
            created_at as detected_at,
            message,
            name,
            resource_type,
            model_execution_id
        FROM {self.elementary_schema}.dbt_run_results
        WHERE created_at >= CURRENT_DATE - 30
        ORDER BY created_at DESC
        """
        return self._execute_query(query, "run_results")
    
    def extract_invocations(self) -> pd.DataFrame:
        """Extract dbt invocation metadata."""
        query = f"""
        SELECT 
            invocation_id,
            job_name,
            command,
            dbt_version,
            run_started_at as invocation_time,
            full_refresh as is_full_refresh,
            invocation_vars as env_vars,
            generated_at,
            job_id,
            job_run_id,
            run_completed_at,
            created_at,
            target_name,
            project_name,
            dbt_user
        FROM {self.elementary_schema}.dbt_invocations
        WHERE run_started_at >= CURRENT_DATE - 30
        ORDER BY run_started_at DESC
        """
        return self._execute_query(query, "invocations")
    
    def extract_model_columns(self) -> pd.DataFrame:
        """Extract model column metadata. Note: model_columns table doesn't exist in this schema."""
        print("⚠️ model_columns table not available in this Elementary schema")
        return pd.DataFrame()
    
    def extract_all_data(self) -> Dict[str, pd.DataFrame]:
        """Extract all Elementary data."""
        if not self.connection:
            self.connect()
        
        try:
            data = {
                'models': self.extract_models(),
                'tests': self.extract_tests(),
                'sources': self.extract_sources(),
                'test_results': self.extract_test_results(),
                'run_results': self.extract_run_results(),
                'invocations': self.extract_invocations(),
                'model_columns': self.extract_model_columns()
            }
            
            print(f"📊 Extracted data summary:")
            for table_name, df in data.items():
                print(f"  - {table_name}: {len(df)} rows")
            
            return data
            
        finally:
            self.disconnect()
    
    def _execute_query(self, query: str, table_name: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        try:
            df = pd.read_sql(query, self.engine)
            # Convert column names to lowercase for consistency with graph builder
            df.columns = df.columns.str.lower()
            print(f"✅ Extracted {len(df)} rows from {table_name}")
            return df
        except Exception as e:
            print(f"❌ Failed to extract {table_name}: {e}")
            return pd.DataFrame()
    
    def _detect_elementary_schema(self) -> str:
        """Auto-detect the elementary schema."""
        if not self.connection:
            raise RuntimeError("Must be connected to detect elementary schema")
        
        # Common elementary schema patterns
        base_schema = self.config.get('schema', '')
        possible_schemas = [
            f"{base_schema}_elementary",
            "elementary",
            f"{base_schema.upper()}_ELEMENTARY",
            "ELEMENTARY"
        ]
        
        cursor = self.connection.cursor()
        
        for schema in possible_schemas:
            try:
                # Test if dbt_models table exists in this schema
                test_query = f"SELECT 1 FROM {schema}.dbt_models LIMIT 1"
                cursor.execute(test_query)
                cursor.fetchone()
                
                print(f"✅ Found Elementary tables in schema: {schema}")
                cursor.close()
                return schema
                
            except Exception:
                continue
        
        cursor.close()
        
        # Fallback: try to find any schema with elementary tables
        try:
            cursor = self.connection.cursor()
            database = self.config.get('database', 'UNKNOWN')
            
            # Query information schema to find elementary tables
            info_query = f"""
            SELECT DISTINCT table_schema
            FROM {database}.information_schema.tables
            WHERE table_name = 'DBT_MODELS'
            AND table_schema ILIKE '%elementary%'
            LIMIT 1
            """
            
            cursor.execute(info_query)
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                schema = result[0]
                print(f"✅ Auto-detected Elementary schema: {schema}")
                return schema
                
        except Exception as e:
            print(f"⚠️ Could not auto-detect elementary schema: {e}")
        
        # Final fallback
        fallback_schema = f"{base_schema}_elementary" if base_schema else "elementary"
        print(f"⚠️ Using fallback elementary schema: {fallback_schema}")
        return fallback_schema
    
    def set_elementary_schema(self, schema: str) -> None:
        """Manually set the elementary schema."""
        self.elementary_schema = schema
        print(f"📝 Elementary schema set to: {schema}")
    
    def parse_json_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Parse JSON string column into structured data."""
        if column in df.columns:
            df[f"{column}_parsed"] = df[column].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else []
            )
        return df