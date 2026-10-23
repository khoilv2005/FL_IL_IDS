"""Metadata-only Task0 partition audit. No training, raw data or TEST reads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fed_learning.strategies.decentralized.denice_v2.partition import bootstrap_similarity, partition_clients


def inspect_manifest(manifest, *, current_classes, manifest_sha256, threshold=.8):
    if not manifest.get('completed') or not manifest.get('content_disjoint'):
        raise ValueError('Requires a completed clean BASE role manifest')
    classes = tuple(sorted(current_classes))
    if not classes or len(set(classes)) != len(classes) or any(type(c) is not int or c < 0 for c in classes):
        raise ValueError('Invalid explicit Task0 class mapping')
    eligible = manifest['eligible_base_experts']
    active = sorted({int(cid) for c in classes for cid in eligible.get(str(c), [])})
    counts = {}
    for cid in active:
        row = manifest['clients'][str(cid)]['role_class_counts']['base']
        counts[cid] = {c: int(row.get(str(c), 0)) for c in classes if int(row.get(str(c), 0)) > 0}
    similarities = bootstrap_similarity(counts, data_task=0, task_classes=classes,
        reference_hash='dry_run_common_task0_initialization:' + manifest_sha256)
    partition = partition_clients(similarities, threshold=threshold)
    clusters, gaps = [], []
    for index, members in enumerate(partition.clusters):
        union = sorted({c for cid in members for c in counts[cid]})
        local_gaps = []
        for cid in members:
            for c in classes:
                if c not in counts[cid]:
                    suppliers = [peer for peer in members if peer != cid and c in counts[peer]]
                    item = dict(client_id=cid, class_id=c, peer_owners=suppliers)
                    gaps.append(item); local_gaps.append(item)
        clusters.append(dict(cluster=index, members=list(members), member_count=len(members),
            class_union=union, missing_current_classes=sorted(set(classes) - set(union)),
            local_data_gap_pairs=len(local_gaps), peer_fillable_data_gap_pairs=sum(bool(g['peer_owners']) for g in local_gaps)))
    fillable = sum(bool(g['peer_owners']) for g in gaps)
    return dict(completed=True, task=0, current_classes=list(classes), threshold=threshold,
        manifest_sha256=manifest_sha256, similarity_hash=partition.similarity_hash,
        source='current_BASE_class_support_metadata', state_compatibility='assumed shared fresh Task0 initialization only',
        future_class_counts_used=False, raw_data_read=False, test_read=False, training=False,
        active_clients=len(active), dormant_clients=len(manifest['clients']) - len(active),
        K=partition.K, singleton_clusters=sum(len(g) == 1 for g in partition.clusters),
        full_current_class_coverage_clusters=sum(not c['missing_current_classes'] for c in clusters),
        local_data_gap_pairs=len(gaps), peer_fillable_data_gap_pairs=fillable,
        peer_fillable_fraction=fillable / len(gaps) if gaps else None,
        clusters=clusters, data_gaps=gaps,
        interpretation='Data support coverage only, not functional competence or an accuracy result')


def run(source, *, classes, out, threshold=.8):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError('Use a new immutable dry-run output path')
    if source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as archive:
            paths = [n for n in archive.namelist() if n.endswith('role_manifest.json')]
            if len(paths) != 1:
                raise ValueError('Expected exactly one role manifest in archive')
            path = paths[0]
            raw = archive.read(path)
            lock = json.loads(archive.read(path.removesuffix('role_manifest.json') + 'role_lock.json'))
    else:
        raw = source.read_bytes()
        lock = json.loads(source.with_name('role_lock.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(raw).hexdigest()
    if lock.get('manifest_sha256') != digest or not lock.get('locked_before_backbone_training'):
        raise ValueError('Role manifest lock mismatch')
    result = inspect_manifest(json.loads(raw), current_classes=classes, manifest_sha256=digest, threshold=threshold)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('clusters', 'data_gaps')}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Results ZIP or role_manifest.json')
    parser.add_argument('--classes', required=True, type=int, nargs='+', help='Explicit Task0 classes only')
    parser.add_argument('--threshold', type=float, default=.8)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.source, classes=args.classes, threshold=args.threshold, out=args.out)
