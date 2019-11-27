import os
import sys

# project root dir, two directories up
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEBUGGING = True if "pydevd" in sys.modules else False
