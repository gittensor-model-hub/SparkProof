"""Multi-layer decontamination — block eval leakage into training data."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from sparkproof.triton_dataset.task_policy import FORBIDDEN_TRAINING_ORIGINS, assert_trainable_task


class TritonASTCanonicalizer(ast.NodeTransformer):
    """Strip user-defined names while preserving operators, control flow, and Triton APIs."""

    _PRESERVED_NAMES = frozenset({"tl", "triton", "torch"})

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id not in self._PRESERVED_NAMES and node.id not in {"True", "False", "None"}:
            node.id = "_"
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = "_"
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = "_"
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.name = "_"
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.name = "_"
        return self.generic_visit(node)


def get_canonical_structure(code: str) -> str:
    try:
        tree = ast.parse(code)
        canonical = TritonASTCanonicalizer().visit(tree)
        ast.fix_missing_locations(canonical)
        return ast.dump(canonical, annotate_fields=True, include_attributes=False)
    except SyntaxError:
        return f"syntax-error:{hashlib.sha256(code.encode()).hexdigest()}"


def text_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()


def row_gpu_architecture(task_or_row: dict[str, Any]) -> str:
    """Architecture label for dataset dedupe. Defaults to blackwell for legacy rows."""
    meta = task_or_row
    if "metadata" in task_or_row:
        meta = (task_or_row.get("metadata") or {}).get("prompt_meta") or task_or_row
    return str(
        task_or_row.get("gpu_architecture")
        or meta.get("gpu_architecture")
        or "blackwell"
    )


def semantic_task_fingerprint(task: dict[str, Any]) -> str:
    """Normalized metadata fingerprint for near-duplicate task detection.

    `target` defaults to "blackwell" for rows with no `gpu_architecture` (all
    pre-existing corpora), so fingerprints for that default population are
    unchanged. Rows targeting a different architecture get a distinct
    fingerprint even if otherwise identical — they are legitimately different
    training examples, not duplicates.
    """
    payload = {
        "op": task.get("task_family") or task.get("category"),
        "target_api": task.get("target_api"),
        "torch_reference": task.get("torch_reference"),
        "dtype": task.get("dtype"),
        "shape_class": task.get("shape_class"),
        "layout": task.get("layout"),
        "target": task.get("gpu_architecture", "blackwell"),
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

    def __init__(
        self,
        *,
        problems_dir: Path | None = None,
        benchmark_py_dir: Path | None = None,
        require_eval_corpus: bool = False,
    ) -> None:
        self.structures: set[str] = set()
        self.prompt_hashes: set[str] = set()
        self.semantic_hashes: set[str] = set()
        self._load_eval_yaml(problems_dir)
        if benchmark_py_dir is None:
            configured_py_dir = os.environ.get("SPARKPROOF_TRITONBENCH_PY_DIR")
            if configured_py_dir:
                benchmark_py_dir = Path(configured_py_dir).expanduser()
        if benchmark_py_dir and benchmark_py_dir.exists():
            self._load_py_tree(benchmark_py_dir)
        if require_eval_corpus and not self.prompt_hashes:
            raise RuntimeError(
                "decontamination requires a TritonBench problem corpus; pass problems_dir "
                "or set SPARKPROOF_TRITONBENCH_PROBLEMS"
            )

    @property
    def fingerprint_counts(self) -> dict[str, int]:
        return {
            "prompts": len(self.prompt_hashes),
            "semantics": len(self.semantic_hashes),
            "structures": len(self.structures),
        }

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

    def add_eval_pytorch_code(self, code: str) -> None:
        """Fingerprint a held-out PyTorch problem (e.g. KernelBench) for decontam."""
        text = code.strip()
        if not text:
            return
        self.prompt_hashes.add(text_fingerprint(text))
        self.structures.add(get_canonical_structure(text))

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
        torch_ref = str(task.get("torch_reference") or task.get("reference_expr") or "").strip()
        if torch_ref:
            if text_fingerprint(torch_ref) in self.prompt_hashes:
                issues.append("torch_reference matches eval fingerprint")
            if self.is_contaminated_code(torch_ref):
                issues.append("torch_reference structure matches eval corpus")
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


def filter_decontaminated(
    records: list[dict[str, Any]],
    problems_dir: Path | None = None,
    *,
    require_eval_corpus: bool = True,
) -> list[dict[str, Any]]:
    return TritonDecontaminator(
        problems_dir=problems_dir,
        require_eval_corpus=require_eval_corpus,
    ).filter_trajectories(records)


def assert_trainable_prompt_record(record: dict[str, Any]) -> None:
    assert_trainable_task(record)
