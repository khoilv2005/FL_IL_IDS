"""Bounded compatibility matrix, real provider calls, no fallback models."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(r'E:\Study\SE373\Project\vuongdat67\novus')
OUT = ROOT / 'reports/codex-test'
source = Path(__file__).with_name('check_cloudflare_live.py').read_text()
catalog = json.loads((OUT / 'cloudflare-catalog.json').read_text())
results = []
report = {'started_at':datetime.now(timezone.utc).isoformat(), 'scope':'Text generation models; synthetic JSON and prescribed plan compatibility, not vulnerability detection accuracy', 'results':results}

def save():
    (OUT / 'cloudflare-model-matrix.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Cloudflare / NOVUS compatibility checks', '', report['scope'], '', '| Model | Direct JSON | NOVUS adapter | Seconds |', '|---|---|---|---|---|']
    for row in results:
        tests = row.get('tests', [])
        def status(t):
            if t.get('passed'): return 'PASS'
            if t.get('skipped'): return 'SKIP'
            return str(t.get('status_code') or t.get('error_type') or 'FAIL')
        lines.append('| '+row['model']+' | '+(status(tests[0]) if tests else row.get('reason','SKIP'))+' | '+(status(tests[1]) if len(tests)>1 else 'SKIP')+' | '+str(round(sum(t.get('seconds',0) for t in tests),2))+' |')
    (OUT / 'cloudflare-model-matrix.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

for row in catalog:
    if row.get('task', {}).get('name') != 'Text Generation': continue
    model = row['name']
    if model.startswith('@cf/cloudflare/') or 'llama-guard' in model:
        results.append({'model':model,'reason':'Specialized decision/moderation model; not a NOVUS chat planner'})
        save()
        continue
    previous = {'@cf/zai-org/glm-5.3':'cloudflare-live.json', '@cf/meta/llama-3.3-70b-instruct-fp8-fast':'cloudflare-llama-live.json'}.get(model)
    if previous:
        result = json.loads((OUT / previous).read_text())
        result['reused_previous_test'] = True
    else:
        print('TEST_MODEL ' + model, flush=True)
        filename = 'cf-' + model.split('/')[-1] + '.json'
        code = source.replace("MODEL = config.get('CLOUDFLARE_MODEL') or '@cf/zai-org/glm-5.3'", 'MODEL = '+repr(model))
        code = code.replace("reasoning_effort='low', ", '').replace('timeout=60', 'timeout=30').replace('max_tokens=2048','max_tokens=1500')
        code = code.replace('cloudflare-live.json',filename)
        exec(compile(code, 'bounded_cloudflare_check', 'exec'), {'__name__':'__main__'})
        result = json.loads((OUT / filename).read_text())
    results.append(result)
    save()
    if any(t.get('status_code') == 429 for t in result.get('tests', [])):
        report['stopped_reason'] = 'HTTP 429: quota or rate limit; remaining models not called'
        save()
        break
print('MATRIX_COMPLETE ' + str(len(results)), flush=True)
