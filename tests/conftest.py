"""
Pytest conftest - adds project root to sys.path and tests/ for helper imports.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add tests/ directory so 'from helpers import ...' works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
