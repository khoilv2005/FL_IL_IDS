"""Generate the Kaggle class-28 diagnostic notebook from its standalone entry."""
import ast
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    code=(root/'debug_denice_class28_kaggle.py').read_text(encoding='utf-8');ast.parse(code)
    notebook=dict(cells=[dict(cell_type='markdown',metadata={},source=[
        '# Debug class 28 on clean results (8)\n',
        'Enable GPU/Internet and attach the original 100-client dataset. Set RESULTS_DRIVE_URL in the code cell.\n',
        'Downloads results (8), extracts only role artifacts and task 4/5 archives. No backbone retraining or final-test reads.\n',
        'Fits balanced multiclass routers from checkpoint BASE memory; compares normal and Oracle-task-4 predictions on saved META-fit/validation inputs.\n',
        'Self + 16 and all-positive-alpha peer coverage are labeled diagnostics, not deployable oracle performance.\n',
        'Output: denice_class28_diagnostics.zip; completion.json must have completed=true.\n']),
        dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),
        nbformat=4,nbformat_minor=5)
    (root/'debug_denice_class28_kaggle.ipynb').write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')


if __name__=='__main__':main()
