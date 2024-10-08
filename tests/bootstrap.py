# bootstrap.py

import sys
import os

def set_paths():
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    print("project_root:", project_root)
    sys.path.insert(0, project_root)

