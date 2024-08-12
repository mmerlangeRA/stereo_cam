import os
import sys

print("hello from init")
def set_paths():
    project_root = os.path.dirname(__file__)
    print("project_root:", project_root)
    sys.path.insert(0, project_root)



set_paths()