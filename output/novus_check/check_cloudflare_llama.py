from pathlib import Path

p = Path(__file__).with_name('check_cloudflare_live.py')
s = p.read_text()
s = s.replace("MODEL = config.get('CLOUDFLARE_MODEL') or '@cf/zai-org/glm-5.3'", "MODEL = '@cf/meta/llama-3.3-70b-instruct-fp8-fast'")
s = s.replace("reasoning_effort='low', ", '')
s = s.replace('cloudflare-live.json', 'cloudflare-llama-live.json')
exec(compile(s, str(p), 'exec'))
