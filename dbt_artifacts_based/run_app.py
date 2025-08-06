#!/usr/bin/env python3
"""
Simple launcher script for the dbt Impact Analysis Streamlit app.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    app_file = Path(__file__).parent / "impact_analysis_app.py"
    
    if not app_file.exists():
        print(f"Error: App file not found at {app_file}")
        sys.exit(1)
    
    # Check if data files exist
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists() or not (data_dir / "knowledge_graph.gpickle").exists():
        print("Error: Graph data not found. Please run the manifest parser first:")
        print("  python graph_storage.py path/to/manifest.json --storage-dir data")
        sys.exit(1)
    
    print("🚀 Starting dbt Impact Analysis Streamlit app...")
    print("📊 App will be available at: http://localhost:8502")
    print("🛑 Press Ctrl+C to stop the app")
    
    try:
        subprocess.run([
            "streamlit", "run", str(app_file),
            "--server.port", "8502",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped.")
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install it:")
        print("  pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()