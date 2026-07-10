"""Multi-layer decontamination — block eval leakage into training data."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sparkproof.triton_dataset.task_policy import FORBIDDEN_TRAINING_ORIGINS, assert_trainable_task


class TritonASTHasher(ast.NodeVisitor):
    """Canonical structural fingerprint (names stripped)."""

    def __init__(self) -> None:
        self.struct_lines: list[str] = []

    def visit_Name(self, node: ast.Name) -> None:
        self.struct_lines.append("Name")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.struct_lines.append(f"FuncDef(args={len(node.args.args)})")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "tl":
                self.struct_lines.append(f"Call:tl.{node.func.attr}")
            elif node.func.value.id == "triton":
                self.struct_lines.append(f"Call:triton.{node.func.attr}")
            else:
                self.struct_lines.append("Call")
        else:
            self.struct_lines.append("Call")
        self.generic_visit(node)


def get_canonical_structure(code: str) -> str:
    try:
        tree = ast.parse(code)
        hasher = TritonASTHasher()
        hasher.visit(tree)
        return "\n".join(hasher.struct_lines)
    except SyntaxError:
        return hashlib.sha256(code.encode()).hexdigest()


def text_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()


def semantic_task_fingerprint(task: dict[str, Any]) -> str:
    """Normalized metadata fingerprint for near-duplicate task detection."""
    payload = {
        "op": task.get("task_family") or task.get("category"),
        "target_api": task.get("target_api"),
        "torch_reference": task.get("torch_reference"),
        "dtype": task.get("dtype"),
        "shape_class": task.get("shape_class"),
        "layout": task.get("layout"),
        "target": "blackwell",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def extract_python_from_response(response: str) -> str:
    for pattern in (r"```python\n(.*?)```", r"```\n(.*?)```"):
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return "\n\n".join(matches)
    return response


class TritonDecontaminator:
    """Pre-load eval fingerprints; reject training rows that overlap."""

    def __init__(self, *, problems_dir: Path | None = None, benchmark_py_dir: Path | None = None) -> None:
        self.structures: set[str] = set()
        self.prompt_hashes: set[str] = set()
        self.semantic_hashes: set[str] = set()
        self._load_eval_yaml(problems_dir)
        if benchmark_py_dir and benchmark_py_dir.exists():
            self._load_py_tree(benchmark_py_dir)

    def _load_eval_yaml(self, problems_dir: Path | None) -> None:
        from sparkproof.triton_dataset.eval_problems import iter_eval_problem_prompts

        for rec in iter_eval_problem_prompts(problems_dir=problems_dir):
            self.prompt_hashes.add(text_fingerprint(rec.get("prompt", "")))
            self.semantic_hashes.add(semantic_task_fingerprint(rec))
            for key in ("ground_truth_code", "broken_code"):
                code = rec.get(key) or ""
                if code.strip():
                    self.structures.add(get_canonical_structure(code))

    def _load_py_tree(self, path: Path) -> None:
        for f in path.rglob("*.py"):
            try:
                self.structures.add(get_canonical_structure(f.read_text()))
            except OSError:
                continue

    def check_task(self, task: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        origin = task.get("origin") or task.get("source")
        if origin in FORBIDDEN_TRAINING_ORIGINS:
            issues.append(f"forbidden origin {origin!r}")
        if task.get("split") in {"test", "eval"}:
            issues.append(f"forbidden split {task.get('split')!r}")
        ph = text_fingerprint(task.get("prompt", ""))
        if ph in self.prompt_hashes:
            issues.append("prompt matches eval fingerprint")
        sh = semantic_task_fingerprint(task)
        if sh in self.semantic_hashes:
            issues.append("semantic fingerprint matches eval task")
        return issues

    def is_contaminated_code(self, code: str) -> bool:
        if not code.strip():
            return True
        return get_canonical_structure(code) in self.structures

    def filter_trajectories(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for rec in records:
            meta = (rec.get("metadata") or {}).get("prompt_meta") or {}
            if self.check_task(meta):
                continue
            code = extract_python_from_response(rec.get("response", ""))
            if self.is_contaminated_code(code):
                continue
            kept.append(rec)
        return kept


def filter_decontaminated(records: list[dict[str, Any]], problems_dir: Path | None = None) -> list[dict[str, Any]]:
    return TritonDecontaminator(problems_dir=problems_dir).filter_trajectories(records)


def assert_trainable_prompt_record(record: dict[str, Any]) -> None:
    assert_trainable_task(record)
