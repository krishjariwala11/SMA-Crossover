#!/usr/bin/env python3
"""
Simple launcher script for the SMA Crossover Strategy application.
This script provides an easy way to run the Streamlit app with proper error handling.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    print("🚀 Launching SMA Crossover Strategy Application...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit is not installed. Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run: pip install -r requirements.txt")
            return
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please ensure you're in the correct directory.")
        return
    
    # Launch the application
    print("🌐 Starting Streamlit application...")
    print("📱 The app will open in your default browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
    except FileNotFoundError:
        print("❌ Streamlit command not found. Please ensure it's properly installed.")

if __name__ == "__main__":
    main()
