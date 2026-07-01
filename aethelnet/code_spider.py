import os
import ast
import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger("CodeSpider")

class CodeSpider:
    """
    The Blueprint Engine:
    Ingests the local codebase to create a structural self-awareness graph.
    Nodes = Files, Classes, Functions.
    Edges = Imports, Calls, Dependencies.
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.nodes = []
        self.edges = []
        self.file_nodes = set()

    def _get_relative_path(self, filepath: str) -> str:
        return os.path.relpath(filepath, self.root_dir)

    def crawl_python_file(self, filepath: str):
        rel_path = self._get_relative_path(filepath)
        self.nodes.append({
            "id": rel_path,
            "type": "file",
            "group": "backend",
            "label": os.path.basename(filepath)
        })
        self.file_nodes.add(rel_path)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content, filename=filepath)
            
            for node in ast.walk(tree):
                # Import tracking
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = alias.name
                        self.edges.append({"source": rel_path, "target": target, "type": "import"})
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        target = node.module
                        self.edges.append({"source": rel_path, "target": target, "type": "import_from"})
                
                # Class / Function tracking (optional for deeper granularity)
                elif isinstance(node, ast.ClassDef):
                    class_id = f"{rel_path}::{node.name}"
                    self.nodes.append({"id": class_id, "type": "class", "group": "backend", "label": node.name})
                    self.edges.append({"source": rel_path, "target": class_id, "type": "contains"})
                    
        except SyntaxError:
            logger.warning(f"SyntaxError parsing {rel_path}")
        except Exception as e:
            logger.error(f"Failed to parse {rel_path}: {e}")

    def crawl_vue_file(self, filepath: str):
        rel_path = self._get_relative_path(filepath)
        self.nodes.append({
            "id": rel_path,
            "type": "file",
            "group": "frontend",
            "label": os.path.basename(filepath)
        })
        self.file_nodes.add(rel_path)
        
        # Simple regex/text search for Vue imports
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if "import " in line and "from" in line:
                        parts = line.split("from")
                        if len(parts) > 1:
                            target = parts[1].strip().strip("'").strip('"').strip(';')
                            self.edges.append({"source": rel_path, "target": target, "type": "import"})
        except Exception as e:
            logger.error(f"Failed to parse {rel_path}: {e}")

    def execute_crawl(self) -> Dict[str, Any]:
        logger.info(f"🕸️ CodeSpider deploying web over repository: {self.root_dir}")
        for root, dirs, files in os.walk(self.root_dir):
            # Exclude node_modules, .git, venv, __pycache__
            dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', 'venv', '.venv', '__pycache__', '.gemini', 'dist')]
            
            for file in files:
                filepath = os.path.join(root, file)
                if file.endswith(".py"):
                    self.crawl_python_file(filepath)
                elif file.endswith(".vue"):
                    self.crawl_vue_file(filepath)
                    
        # Cleanup edges to only link to known internal nodes if possible, or keep external for context
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "metadata": {"total_files": len(self.file_nodes)}
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spider = CodeSpider(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    graph = spider.execute_crawl()
    
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../blueprint_graph.json"))
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"Blueprint graph generated with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.")
    print(f"Saved to {out_path}")
