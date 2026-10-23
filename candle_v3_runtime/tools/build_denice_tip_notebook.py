"""Build the frozen-router Kaggle notebook from the existing download setup."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, text):
    result = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main():
    original = json.loads((ROOT/'eval_denice_oracle_results4_kaggle.ipynb').read_text(encoding='utf-8'))
    setup = ''.join(original['cells'][1]['source'])
    setup = setup.replace('denice_oracle_03b9b53', 'denice_tip_03b9b53')
    setup = setup.replace('denice_oracle_source', 'denice_tip_source')
    setup = setup.replace('BATCH_SIZE = 5, 19, 50000, 1024', 'BATCH_SIZE = 5, 19, 50000, 512')
    setup = setup.replace('source_commit =', "else:\n    subprocess.run(['git','-C',str(SOURCE),'pull','--ff-only'],check=True)\nsource_commit =")
    download = ''.join(original['cells'][2]['source'])
    prepare = '''from tools.eval_denice_tip_router import run_suite, write_json
from threadpoolctl import threadpool_limits
import hashlib
loader = IncrementalDataLoader(DATA_DIR)
classes = {t:list(map(int,loader.get_task_classes(t))) for t in range(TASK+1)}
seed = int(config.get('random_seed',config.get('seed',42)))
x,y = loader.get_test_data(TASK,cumulative=True)
# Carry original test-row identities through exactly the existing samplers.
indices, selected_y, sample_info = _limit_eval_samples(
    torch.arange(len(y))[:,None], y, MAX_SAMPLES, seed+TASK)
index_shards, partition = _partition_test_data_by_client(
    indices, selected_y, ids, seed+104729*TASK)
shards = {}
for cid, shard in index_shards.items():
    selected = shard['X_test'].reshape(-1).long()
    shards[cid] = dict(X_test=x[selected],y_test=shard['y_test'],sample_ids=selected.numpy())
loader._test_data = None
del x,y,indices,index_shards
gc.collect()
digest = hashlib.sha256()
with (root/name).open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):
        digest.update(block)
protocol = dict(
    kind='retrospective historical-local-train router refit; not streaming replay-free',
    training_commit=config['git_commit'],evaluation_commit=source_commit,
    fedprotip_reference_commit='54193fa2d44f6203f39299a0ac3845097559a440',
    checkpoint=name,checkpoint_file_sha256=digest.hexdigest(),
    data_dir=DATA_DIR,seed=seed,task_classes=classes,client_ids=ids,
    sample_info=sample_info,partition=partition,feature='adapter-free fc1; identity transform',
    profile_max_samples=512,profile_validation_fraction=0.2,
    selection='per-client task-macro accuracy on router validation; fixed candidate grid',
    primary_gate='mean-client accuracy gain >=2pp; pooled macro-F1 drop <=0.5pp; paired client CI low >0',
    policies={
      'Router':'saved binary cosine; local class mask',
      'Multiclass':'balanced logistic regression on saved binary memory; local class mask',
      'Centroid':'continuous nearest centroid; local class mask',
      'Mahalanobis':'continuous pooled shrinkage covariance; local class mask',
      'TIP':'per-sample subspace reference matching; local class mask',
      'OracleLocal':'true task adapter; original local mask including existing fallback',
      'OracleGlobal':'true task adapter; global task mask',
      'AllClasses':'no adapter; all seen classes'},
    units='accuracy and F1 as fractions; deltas and confidence intervals in percentage points')
write_json(OUT/'protocol.json',protocol)
print('Evaluation only. No training, no peer TIP, no graph changes.')
print('Output:',OUT)
'''
    evaluate = '''# P0 fails early if legacy accuracy differs by more than 0.1 percentage point.
# P1 fits on local train data only; P2 uses exactly the original 50,000 test shards.
with threadpool_limits(limits=1):
    summary = run_suite(ckpt,config,report,loader,shards,classes,ids,OUT,device,
                        batch_size=BATCH_SIZE,max_profile_samples=512,seed=seed)
display(summary)
print((OUT/'gate.json').read_text())
archive = Path('/kaggle/working/denice_tip_diagnostics.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file() and 'checkpoints' not in p.relative_to(OUT).parts:
            z.write(p,str(p.relative_to(OUT)))
print('Download:',archive)
'''
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=original['metadata'], cells=[
        cell('markdown', '# DeNICE frozen router diagnostics — P0 → P1 → P2\n\n'
             'Upload this notebook to Kaggle, attach the original 100-clients dataset and enable Internet. '
             'It downloads results (4).zip from the configured Drive link and restores checkpoint 03b9b53. '
             '**Evaluation/profile fitting only; no training.** GPU is optional.\n\n'
             'Compares legacy, multiclass, continuous centroid, Mahalanobis, TIP, OracleLocal, '
             'OracleGlobal and AllClasses. Historical train access is retrospective, not a streaming claim. '
             'The baseline guard deliberately stops incompatible runs.\n'),
        cell('code',setup), cell('code',download), cell('code',prepare), cell('code',evaluate)])
    for index, item in enumerate(notebook['cells']):
        item['id'] = f'tip-{index}'
    path = ROOT/'eval_denice_tip_router_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
