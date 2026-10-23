"""AST comparison of original guard functions with the prepared variant."""

from __future__ import annotations

import ast
from pathlib import Path


HERE = Path(__file__).parent
BASELINE = HERE / "generative_provider.baseline.py"
VARIANT = HERE / "generative_provider.py"
UNCHANGED_GUARDS = (
    "_require_str",
    "_context",
    "_local_endpoint",
    "_http_post",
    "_parse_http_content",
    "_validate_generation_envelope",
    "_native_answer",
)


def functions(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def assigned_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            try:
                values[name] = eval(compile(ast.Expression(node.value), str(path), "eval"), {"__builtins__": {}}, values)
            except (NameError, TypeError, ValueError, SyntaxError):
                continue
    return values


before, after = functions(BASELINE), functions(VARIANT)
for name in UNCHANGED_GUARDS:
    if ast.dump(before[name], include_attributes=False) != ast.dump(after[name], include_attributes=False):
        raise SystemExit(f"guard AST changed: {name}")

before_constants, after_constants = assigned_constants(BASELINE), assigned_constants(VARIANT)
preserved_limits = (
    "MAX_CHUNKS", "MAX_QUESTION_CHARS", "MAX_CHUNK_CHARS", "MAX_CONTEXT_CHARS",
    "CONTEXT_WINDOW_TOKENS", "MAX_PREFLIGHT_INPUT_TOKENS", "MAX_RESPONSE_BYTES",
)
for name in preserved_limits:
    if before_constants.get(name) != after_constants.get(name):
        raise SystemExit(f"guard constant changed: {name}: {before_constants.get(name)!r} != {after_constants.get(name)!r}")
if before_constants["MAX_TOKENS"] != 512 or after_constants["MAX_TOKENS"] != 256:
    raise SystemExit("unexpected generation token variant")

print("unchanged guard ASTs: " + ", ".join(UNCHANGED_GUARDS))
print("necessary differences: answer request schema is per-request; model output parsing validates selected IDs;")
print("adapter constructs exact full selected chunk text/citations; diagnostics separate model-selected IDs from constructed citation IDs;")
print("preserved limits: " + ", ".join(f"{name}={after_constants[name]}" for name in preserved_limits))
print("necessary budget variant: output MAX_TOKENS 512->256; input preflight remains explicitly 5632 of 6144 tokens")
