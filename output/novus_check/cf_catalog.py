import json
from pathlib import Path
import httpx
from dotenv import dotenv_values
root = Path(r'E:\Study\SE373\Project\vuongdat67\novus')
c = dotenv_values(root / '.env')
rows = []
with httpx.Client(timeout=30) as client:
    for page in range(1, 11):
        r = client.get('https://api.cloudflare.com/client/v4/accounts/' + c['CLOUDFLARE_ACCOUNT_ID'] + '/ai/models/search', headers={'Authorization': 'Bearer ' + c['CLOUDFLARE_API_TOKEN']}, params={'page':page,'per_page':100})
        print('catalog_status', r.status_code)
        if r.status_code != 200: break
        data = r.json()
        batch = data.get('result', [])
        rows.extend(batch)
        if len(batch) < 100: break
(root / 'reports/codex-test/cloudflare-catalog.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
for row in rows:
    if 'text generation' in str(row.get('task', '')).lower():
        print(json.dumps({'name':row.get('name'),'task':row.get('task'),'properties':row.get('properties')}))
