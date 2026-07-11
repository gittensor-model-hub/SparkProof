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
        "torch.sigmoid",
        "torch.tanh",
        "torch.nn.functional.softmax",
        "torch.nn.functional.log_softmax",
        "torch.nn.functional.layer_norm",
        "torch.nn.functional.gelu",
        "torch.nn.functional.relu",
        "torch.nn.functional.silu",
        "torch.nn.functional.leaky_relu",
        "torch.nn.functional.softplus",
        "F.softmax",
        "F.matmul",
        "F.layer_norm",
        "F.gelu",
        "F.relu",
        "F.silu",
        "F.sigmoid",
        "F.tanh",
        "F.leaky_relu",
        "F.softplus",
    }
)

# A candidate that measures its own speed can also try to dodge that
# measurement rather than fake correctness: launching work on a non-default
# CUDA stream escapes the harness's implicit synchronization, and
# reassigning the clock/benchmark functions the harness relies on
# (`triton.testing.do_bench`, `torch.cuda.synchronize`/`Event`) fabricates a
# timing number outright. See KernelBench's EVAL.md / kernel_static_checker.py
# for the reward-hacking patterns this mirrors.
STREAM_INJECTION_MARKERS = ("torch.cuda.stream(", "torch.cuda.Stream(", ".record_stream(")
TIMING_PATCH_TARGETS = frozenset({"do_bench", "synchronize", "Event", "elapsed_time"})

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


def _is_pass_only_body(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """A launcher that inherits/wraps a reference and does nothing of its own."""
    body = [
        node
        for node in function.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
    ]
    return len(body) == 1 and isinstance(body[0], ast.Pass)


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
        function = functions[function_name]
        if _is_pass_only_body(function):
            issues.append(f"{function_name} body is only 'pass' (inheritance bypass)")
        if any(isinstance(node, ast.Try) for node in ast.walk(function)):
            issues.append(f"{function_name} contains try/except (potential fallback bypass)")
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node, aliases)
            if name in FORBIDDEN_LAUNCHER_CALLS:
                issues.append(f"forbidden launcher fallback in {function_name}: {name}")

    return issues


def detect_timing_manipulation(code: str) -> list[str]:
    """Flag attempts to dodge trusted timing via streams or a patched clock.

    The harness measures wall-clock via ``triton.testing.do_bench`` in the
    same process as the candidate; launching on a non-default stream escapes
    its implicit synchronization, and reassigning the clock/benchmark
    functions it relies on fabricates the reported number outright.
    """
    issues: list[str] = []
    for marker in STREAM_INJECTION_MARKERS:
        if marker in code:
            issues.append(f"uses non-default CUDA stream ({marker.rstrip('(')}) — can hide async work from timing")
            break
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return issues
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
            if name in TIMING_PATCH_TARGETS:
                issues.append(f"reassigns {name!r} (potential timing/clock monkey-patch)")
    return issues


def analyze_anti_cheat(code: str) -> dict[str, Any]:
    issues = detect_torch_fallbacks(code) + detect_timing_manipulation(code)
    return {"passed": not issues, "issues": issues}
