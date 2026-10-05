"""Generate the standalone Kaggle notebook from the clean training entry point."""
import ast
import json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    code=(root/'train_denice_cgofed_clean_kaggle.py').read_text(encoding='utf-8')
    ast.parse(code)
    notebook=dict(cells=[dict(cell_type='markdown',metadata={},source=[
        '# Clean DeNICE + CGoFed backbone training\n',
        'Enable GPU and Internet; attach the original 100-client dataset.\n',
        'Fresh six-task run on BASE only; xi=0.5. META/VALIDATION roles are reserved before training.\n',
        'Each round is checkpointed and compressed; each completed task is sealed into one ZIP.\n',
        'Only task 5 final round evaluates clean validation; final test is not opened.\n']),
        dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                      language_info=dict(name='python')),
        nbformat=4,nbformat_minor=5)
    (root/'train_denice_cgofed_clean_kaggle.ipynb').write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')

if __name__=='__main__':main()
