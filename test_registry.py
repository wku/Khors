import pathlib
from khors.tools.registry import ToolRegistry

registry = ToolRegistry(repo_dir=pathlib.Path('.').resolve(), drive_root=pathlib.Path('./data').resolve())
print(registry._entries.keys())
