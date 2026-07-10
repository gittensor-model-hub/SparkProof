"""Detect PyTorch fallback cheating in teacher-generated Triton launchers."""

from __future__ import annotations

import ast
from typing import Any

FORBIDDEN_LAUNCHER_CALLS = frozenset(
    {
        "torch.matmul",
        "torch.mm",
        "torch.bmm",
        "torch.softmax",
        "torch.log_softmax",
        "torch.nn.functional.softmax",
        "torch.nn.functional.log_softmax",
        "torch.nn.functional.layer_norm",
        "torch.nn.functional.gelu",
        "torch.nn.functional.relu",
        "torch.nn.functional.silu",
        "F.softmax",
        "F.matmul",
        "F.layer_norm",
        "F.gelu",
        "F.relu",
    }
)

def _call_name(node: ast.Call, aliases: dict[str, str]) -> str | None:
    if isinstance(node.func, ast.Attribute):
        parts: list[str] = []
        current: ast.AST = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            name = ".".join(reversed(parts))
            root, *tail = name.split(".")
            return ".".join([aliases.get(root, root), *tail])
    if isinstance(node.func, ast.Name):
        return aliases.get(node.func.id, node.func.id)
    return None


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _is_triton_jit(node: ast.FunctionDef | ast.AsyncFunctionDef, aliases: dict[str, str]) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Attribute) and isinstance(decorator.value, ast.Name):
            root = aliases.get(decorator.value.id, decorator.value.id)
            if root == "triton" and decorator.attr == "jit":
                return True
    return False


def _launched_kernel(call: ast.Call, jit_names: set[str]) -> str | None:
    if not isinstance(call.func, ast.Subscript):
        return None
    target = call.func.value
    if isinstance(target, ast.Name) and target.id in jit_names:
        return target.id
    return None


def detect_torch_fallbacks(code: str) -> list[str]:
    """Flag compute fallbacks reachable from a custom-kernel launcher.

    Top-level self-tests may use PyTorch reference operators. Only launcher
    functions and their local helper call graph are inspected for fallbacks.
    """
    issues: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["launcher syntax error"]

    aliases = _import_aliases(tree)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    jit_names = {name for name, node in functions.items() if _is_triton_jit(node, aliases)}

    if not jit_names:
        issues.append("no @triton.jit kernel found")
        return issues

    launchers: set[str] = set()
    for name, function in functions.items():
        if name in jit_names:
            continue
        if any(
            _launched_kernel(node, jit_names)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        ):
            launchers.add(name)

    if not launchers:
        issues.append("no kernel grid launch detected")
        return issues

    reachable = set(launchers)
    pending = list(launchers)
    while pending:
        function = functions[pending.pop()]
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node, aliases)
            if name in functions and name not in jit_names and name not in reachable:
                reachable.add(name)
                pending.append(name)

    for function_name in sorted(reachable):
        for node in ast.walk(functions[function_name]):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node, aliases)
            if name in FORBIDDEN_LAUNCHER_CALLS:
                issues.append(f"forbidden launcher fallback in {function_name}: {name}")

    return issues


def analyze_anti_cheat(code: str) -> dict[str, Any]:
    issues = detect_torch_fallbacks(code)
    return {"passed": not issues, "issues": issues}
