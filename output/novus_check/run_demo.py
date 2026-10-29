"""Run the user's novus demo and benchmark on an ephemeral loopback port."""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

import httpx
from agentic_pentest.presentation.demo_server import start_live_demo_server
from agentic_pentest.application.use_cases.eval_runner import BenchmarkEvaluationRunner
from agentic_pentest.application.use_cases.verify_differential import DifferentialVerifier
from agentic_pentest.domain.entities import HTTPMethod, Principal, Role, TargetObject
from agentic_pentest.infrastructure.executors.httpx_executor import HTTPXExecutor


async def main():
    server = start_live_demo_server(port=0)
    target = f'http://127.0.0.1:{server.server_port}'
    out = Path('reports/codex-test')
    out.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            health = await client.get(target+'/identity/api/auth/v1/health')
        assert health.status_code == 200
        finding = await DifferentialVerifier(HTTPXExecutor(allowed_hosts=['127.0.0.1'])).verify_bola(
            endpoint_url=target+'/identity/api/v2/vehicle/{id}/location', method=HTTPMethod.GET,
            target_object=TargetObject(id='v-alice-101',type_name='vehicle',owner_principal_id='user_alice_01',value='v-alice-101'),
            legitimate_owner=Principal(id='user_alice_01',name='Alice',role=Role.USER,jwt_token='token_alice'),
            attacker=Principal(id='user_bob_02',name='Bob',role=Role.USER,jwt_token='token_bob'))
        assert finding is not None
        statuses = [finding.triplet.baseline_legitimate[1].status_code,
                    finding.triplet.cross_principal_test[1].status_code,
                    finding.triplet.unauthenticated_control[1].status_code]
        assert statuses == [200,200,401]
        print('LOCAL DEMO: health=200, BOLA verified, triplet='+str(statuses),flush=True)
        report = await BenchmarkEvaluationRunner(target_base_url=target).run_evaluation()
        result = dict(target=target,server_stopped_after_test=True,health=health.status_code,
            bola_verified=True,triplet=statuses,benchmark=asdict(report),live_llm_tested=False)
        (out/'demo-benchmark.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
        print('BENCHMARK',report.passed_count,'/',report.total_questions,'EX',report.accuracy_ex,flush=True)
        for r in report.results:
            if not r.passed:print('FAILED',r.question_id,r.rationale,flush=True)
    finally:
        server.shutdown()
        server.server_close()


asyncio.run(main())
