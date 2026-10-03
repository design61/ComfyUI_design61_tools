from pathlib import Path
import importlib.util
import sys
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ComfyUI_design61_tools", root/"__init__.py", submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
module.__path__ = [str(root)]
