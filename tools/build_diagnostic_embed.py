"""Synchronize the single-cell launcher with the standalone diagnostic worker."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = 'import sys\nsys.path.insert(0, sys.argv[1])\n' + (root/'tools/diagnose_cluster_servers.py').read_text(encoding='utf-8')
entry = '''
from types import SimpleNamespace
from fed_learning.strategies.decentralized.denice_v2.data import TaskData
args = json.loads(sys.argv[2])
assert torch.__version__ == args.pop('expected_torch'), 'Eval runtime mismatch'
args['device'] = args['device'] or ('cuda:0' if torch.cuda.is_available() else 'cpu')
payload = completed_payload(args['checkpoint_root'])
state = restore(payload['states'][next(iter(payload['active']))])
source = SimpleNamespace(dataset=Path(args.pop('data_dir')), cache=Path(args['output_dir']).parent/'test_cache',
    input_shape=tuple(state.algorithm_state['native_input_shape']))
x, y, _ = TaskData.test(source)
del payload, state
result = diagnose(x=x, y=y, **args)
print({k:v for k,v in result.items() if k != 'results'}, flush=True)
'''
path = root/'eval_selective.py'
text = path.read_text(encoding='utf-8')
marker = '# BEGIN EMBEDDED DIAGNOSTIC'
if marker in text:
    text = text[:text.index(marker)] + text[text.index("if __name__ == '__main__':"):]
text = text.replace("if __name__ == '__main__':", marker+'\nDIAGNOSTIC_SOURCE = '+repr(source)+
    '\nDIAGNOSTIC_ENTRY = '+repr(entry)+"\n\nif __name__ == '__main__':")
path.write_text(text, encoding='utf-8')
