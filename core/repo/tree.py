"""Plain-text file tree, easy to show in any UI and easy to test."""

from __future__ import annotations

from collections.abc import Iterable


def render_tree(
    paths: Iterable[str], annotations: dict[str, str] | None = None, root_name: str = "repo"
) -> str:
    """Render forward-slash relative paths as a tree. Folders first, then files, A-Z."""
    notes = annotations or {}
    root: dict = {}
    for path in sorted(paths):
        node = root
        parts = path.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = None  # None marks a file

    lines = [f"{root_name}/"]

    def walk(node: dict, prefix: str, parent: str) -> None:
        names = sorted(node, key=lambda n: (node[n] is None, n.lower()))
        for index, name in enumerate(names):
            last = index == len(names) - 1
            connector = "└── " if last else "├── "
            rel = f"{parent}{name}"
            child = node[name]
            if child is None:
                note = notes.get(rel)
                lines.append(f"{prefix}{connector}{name}" + (f"  {note}" if note else ""))
            else:
                lines.append(f"{prefix}{connector}{name}/")
                walk(child, prefix + ("    " if last else "│   "), rel + "/")

    walk(root, "", "")
    return "\n".join(lines)