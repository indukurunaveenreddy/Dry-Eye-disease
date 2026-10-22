"""
OculoAI • Clinical Dry Eye Suite
Root Entry Point for GitHub Codespaces & Streamlit Community Cloud
"""

import os
import sys
import runpy

# Determine absolute path to the main application directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, 'Dry Eye Disease', 'Source Code', 'Eye DIsease')

if not os.path.exists(APP_DIR):
    # Fallback if launched from inside the subfolder
    APP_DIR = os.path.join(ROOT_DIR, 'Source Code', 'Eye DIsease')

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Switch to the application directory so relative datasets and databases resolve
os.chdir(APP_DIR)

# Execute Login Portal
login_script = os.path.join(APP_DIR, 'Login.py')
runpy.run_path(login_script, run_name='__main__')
