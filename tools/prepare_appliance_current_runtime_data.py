"""Stage all owned BASE/CAL task partitions once for automatic runtime use.

Preparation reads original own shards; runtime never falls back to them. No
validation/test role is opened and staging does not grant training access to
future tasks. The locked current-task provider is still mandatory at runtime.
"""
import argparse,json,sys
from pathlib import Path
from unittest.mock import patch
from appliance.current_base_data import prepare_store as prepare_base
from appliance.current_calibration_data import prepare_store as prepare_cal
from appliance.state import write_json
from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256


def run(a):
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    write_json(a.out/'completion.json',dict(completed=False,stage='preparation'))
    roles=CleanRoleData(a.roles,source_data_dir=a.data_dir)
    clients=sorted(map(int,roles.manifest['clients']))
    original=CleanRoleData.client_role;access=[]
    def owned_prepare(self,cid,role,*args,**kwargs):
        if role not in ('base','calibration'):raise AssertionError('Preparation opened forbidden data role')
        print(f'Preparing current runtime partitions: owner={cid}, role={role}',flush=True)
        x,y,rows=original(self,cid,role,*args,**kwargs)
        access.append(dict(client=int(cid),role=role,rows=len(rows)))
        write_json(a.out/'preparation_access.json',access)
        return x,y,rows
    try:
        with patch.object(CleanRoleData,'client_role',owned_prepare):
            base=prepare_base(a.roles,a.data_dir,a.out/'base_store',clients)
            cal=prepare_cal(a.roles,a.data_dir,a.out/'calibration_store',clients)
        if base['role_manifest_sha256']!=cal['role_manifest_sha256'] or base['metadata_sha256']!=cal['metadata_sha256']:
            raise AssertionError('BASE/CAL provenance mismatch')
        result=dict(completed=True,clients=clients,client_count=len(clients),
            role_manifest_sha256=base['role_manifest_sha256'],metadata_sha256=base['metadata_sha256'],
            base_store_manifest_sha256=file_sha256(a.out/'base_store/base_store_manifest.json'),
            calibration_store_manifest_sha256=file_sha256(a.out/'calibration_store/calibration_store_manifest.json'),
            original_shard_access_is_preparation_only=True,runtime_source_fallback_authorized=False,
            future_runtime_access_authorized=False,validation_opened=False,final_test_opened=False,
            automatic_discovery_verified=False,initial_install_authorized=False)
        write_json(a.out/'completion.json',result);print(json.dumps(result,indent=2),flush=True)
    except Exception as exc:
        write_json(a.out/'completion.json',dict(completed=False,error_type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('roles','data-dir','out'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
