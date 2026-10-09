"""Read-only delivery checks: notebook size/syntax and runtime/source locks."""
import ast
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
notebook = root / 'train_denice_appliance_full_kaggle.ipynb'
value = json.loads(notebook.read_text(encoding='utf-8'))
sources = '\n'.join(''.join(c['source']) for c in value['cells'] if c['cell_type'] == 'code')
ast.parse(sources)
report = json.loads((root / 'artifacts/appliance_production_integration_smoke.json').read_text())
checks = dict(notebook_under_1MB=notebook.stat().st_size < 1_000_000,
    no_embedded_sources='EMBEDDED_' not in sources,
    github_clone_default="'git','clone','--depth','1'" in sources and 'runtime_manifest.json' not in sources,
    integration_pass=report['completed'] and report['mismatches'] == 0,
    source_locks_match=all(hashlib.sha256((root/p).read_bytes()).hexdigest() == h
                          for p, h in report['source_sha256'].items()),
    full_not_started=not report['full_campaign_started'])
if not all(checks.values()):
    raise AssertionError(checks)
print(json.dumps(dict(checks=checks, notebook_bytes=notebook.stat().st_size,
                     integration_checks=report['comparisons']), indent=2))
