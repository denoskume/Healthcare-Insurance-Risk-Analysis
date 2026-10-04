import ast
from pathlib import Path


FILES_REQUIRING_DESCRIPTIVE_NAMES = [
    Path("src/features.py"),
    Path("src/modeling.py"),
    Path("src/evaluation.py"),
    Path("scripts/generate_outputs.py"),
    Path("tests/test_features.py"),
    Path("tests/test_modeling.py"),
    Path("tests/test_evaluation.py"),
]

DISALLOWED_GENERIC_NAMES = {
    "X",
    "y",
    "cv",
    "tr",
    "va",
    "pred",
    "score",
    "rows",
    "fitted",
    "df",
    "col",
    "out",
    "Xtr",
    "Xte",
    "ytr",
    "yte",
    "gtr",
    "gte",
}


def test_core_code_uses_descriptive_variable_names():
    violations = []

    for file_path in FILES_REQUIRING_DESCRIPTIVE_NAMES:
        syntax_tree = ast.parse(file_path.read_text(encoding="utf-8"))
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Name) and node.id in DISALLOWED_GENERIC_NAMES:
                violations.append(f"{file_path}:{node.lineno}:{node.id}")
            if isinstance(node, ast.arg) and node.arg in DISALLOWED_GENERIC_NAMES:
                violations.append(f"{file_path}:{node.lineno}:{node.arg}")

    assert not violations, "Generic names found: " + ", ".join(violations)
