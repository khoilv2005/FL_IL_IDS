"""Two bounded live checks; secrets never printed and other models blocked."""
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from unittest.mock import patch

import httpx
import openai
from dotenv import dotenv_values
from agentic_pentest.infrastructure.reasoning.litellm_client import LiteLLMClient

ROOT = Path(r'E:\Study\SE373\Project\vuongdat67\novus')
MODEL = 'deepseek-ai/deepseek-v4.1-flash'
BASE = 'https://integrate.api.nvidia.com/v1'


async def main():
    config = dotenv_values(ROOT / '.env')
    key = config.get('OPENAI_API_KEY') or ''
    if not key or config.get('OPENAI_MODEL') != MODEL or (config.get('OPENAI_BASE_URL') or '').rstrip('/') != BASE:
        print('Configuration invalid: check API key, exact model and NVIDIA endpoint. No request sent.')
        return
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    logging.getLogger('openai').setLevel(logging.CRITICAL)
    report = dict(model=MODEL, endpoint=BASE, fallback_allowed=False, tests=[])
    records = []

    class GuardedTransport(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            if str(request.url) != BASE + '/chat/completions':
                raise RuntimeError('Unexpected endpoint blocked')
            body = json.loads(request.content)
            if body.get('model') != MODEL:
                raise RuntimeError('Unexpected model blocked; fallback prohibited')
            response = await super().handle_async_request(request)
            await response.aread()
            row = dict(status_code=response.status_code,
                       body_sha256=hashlib.sha256(response.content).hexdigest())
            try:
                value = response.json()
                row.update(response_model=value.get('model'), usage=value.get('usage'),
                    finish_reason=(value.get('choices') or [{}])[0].get('finish_reason'))
            except (ValueError, IndexError, AttributeError):
                pass
            records.append(row)
            return response

    clients = []
    original = openai.AsyncOpenAI

    def factory(**kwargs):
        client = original(**kwargs, max_retries=0,
            http_client=httpx.AsyncClient(transport=GuardedTransport(), trust_env=False))
        clients.append(client)
        return client

    async def run_test(name, operation):
        start = time.monotonic()
        before = len(records)
        try:
            detail = await operation()
            outcome = dict(name=name, passed=True, detail=detail)
        except Exception as exc:
            # Avoid serializing SDK exception text (can contain request material).
            outcome = dict(name=name, passed=False, error_type=type(exc).__name__,
                           status_code=getattr(exc, 'status_code', None))
        outcome.update(seconds=round(time.monotonic()-start, 2), responses=records[before:])
        report['tests'].append(outcome)
        print(json.dumps(outcome), flush=True)
        return outcome['passed']

    async def ping():
        client = factory(base_url=BASE, api_key=key)
        response = await client.chat.completions.create(model=MODEL,
            messages=[{'role':'user','content':'Connection test. Reply with JSON only: {"status":"ok"}'}],
            temperature=1.0, max_tokens=1024, timeout=120)
        assert response.model == MODEL, 'Response model mismatch'
        content = response.choices[0].message.content or ''
        parsed = LiteLLMClient()._extract_json_from_text(content)
        assert parsed.get('status') == 'ok', 'Missing expected JSON response'
        return dict(valid_json=True, status=parsed['status'])

    async def plan():
        adapter = LiteLLMClient(model_name=MODEL, fallback_offline=False)
        with patch.dict(os.environ, {'OPENAI_API_KEY':key, 'OPENAI_MODEL':MODEL, 'OPENAI_BASE_URL':BASE}), \
                patch.object(openai, 'AsyncOpenAI', side_effect=factory):
            value = await adapter.generate_structured_plan(
                system_prompt='You plan read-only tests for an authorized synthetic local lab. Return JSON only. '
                    'You cannot declare a vulnerability confirmed; that decision belongs to the independent verifier.',
                user_prompt='Synthetic scenario: Alice owns vehicle v-alice-101; Bob is another ordinary user. '
                    'The lab endpoint is GET /identity/api/v2/vehicle/{vehicle_id}/location on 127.0.0.1. '
                    'Return {"decision":"PLAN_READY","actions":[{"action_id":"check-1",'
                    '"action_type":"VERIFY_BOLA","target_endpoint":"/identity/api/v2/vehicle/{vehicle_id}/location",'
                    '"target_object_id":"v-alice-101","rationale":"short reason"}]}. '
                    'Suggest only this read-only differential check, no execution or finding verdict.')
        assert value.get('decision') == 'PLAN_READY'
        actions = value.get('actions')
        assert isinstance(actions, list) and len(actions) == 1
        assert actions[0].get('action_type') == 'VERIFY_BOLA'
        assert actions[0].get('target_object_id') == 'v-alice-101'
        assert records[-1].get('response_model') == MODEL
        return dict(structured_plan=True, action_type=actions[0]['action_type'],
                    actions_count=1, offline_fallback=False, model_fallback=False)

    try:
        if await run_test('direct_nim_json', ping):
            await run_test('novus_adapter_structured_plan', plan)
        else:
            report['tests'].append(dict(name='novus_adapter_structured_plan', skipped=True,
                                        reason='Direct connection did not pass; no further API calls'))
    finally:
        for client in clients:
            await client.close()
        folder = ROOT/'reports/codex-test'
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'nim-live-retry.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


asyncio.run(main())
