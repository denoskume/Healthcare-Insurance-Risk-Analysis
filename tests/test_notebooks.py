from pathlib import Path

import nbformat


NOTEBOOK_DIRECTORY = Path("notebooks")


def test_notebooks_are_valid_and_saved_without_execution_errors():
    notebook_paths = sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))

    assert len(notebook_paths) == 7

    for notebook_path in notebook_paths:
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)

        for cell in notebook.cells:
            assert cell.get("id"), f"Missing cell id in {notebook_path}"

            if cell.cell_type != "code":
                continue

            assert cell.execution_count is not None, (
                f"Unexecuted code cell in {notebook_path}"
            )

            execution_errors = [
                output
                for output in cell.outputs
                if output.get("output_type") == "error"
            ]
            assert not execution_errors, (
                f"Saved execution error in {notebook_path}: {execution_errors}"
            )
