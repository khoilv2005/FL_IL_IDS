"""Current-graph, metadata-only candidate selection. No FIT/CAL sample API."""
from itertools import combinations, product
import pickle

import numpy as np

VERSION = 'candle_v4_graph_candidate_preflight_v1'


def positive_neighbors(graph, receiver):
    row=graph.get('alpha_debug',{}).get(receiver,graph.get('alpha_debug',{}).get(str(receiver),{}))
    ids=row.get('group_ids',[]); values=row.get('alphas',[])
    if len(ids)!=len(values) or len(set(map(int,ids)))!=len(ids):raise ValueError('Malformed native neighborhood')
    if any(not np.isfinite(a) or a<0 for a in values):raise ValueError('Invalid native alpha')
    return sorted(int(j) for j,a in zip(ids,values) if int(j)!=receiver and a>0)


def select_fixture(graph, metadata, current_classes, minimum_clients=4, minimum_count=32):
    """First feasible (receiver,c1,c2,d1,d2,filler IDs), never a quality ranking."""
    entries=[]; by_receiver={}
    for receiver in sorted(metadata):
        peers=positive_neighbors(graph,receiver)
        for c in sorted(current_classes):
            if metadata[receiver]['owned_BASE'].get(c,0):continue
            sources=[]
            for donor in peers:
                if donor not in metadata:continue
                row=metadata[donor]
                if (row['current_train_BASE'].get(c,0)>=minimum_count and
                        row['FIT_counts'].get(c,0)>=minimum_count and row['CAL_counts'].get(c,0)>=minimum_count):
                    sources.append(donor)
            entry=dict(receiver=receiver,target=c,eligible_donors=sources,
                       current_peers=peers,quality='UNMEASURED')
            entries.append(entry)
            if sources:by_receiver.setdefault(receiver,{})[c]=sources
    for receiver,candidates in sorted(by_receiver.items()):
        peers=positive_neighbors(graph,receiver)
        for classes in combinations(sorted(candidates),2):
            negative=lambda owner:sum(n for c,n in metadata[owner]['CAL_counts'].items()
                                      if c in current_classes and c not in classes)
            if negative(receiver)<minimum_count:continue
            for donors in product(*(candidates[c] for c in classes)):
                owners=sorted({receiver,*donors})
                if any(negative(o)<minimum_count for o in owners):continue
                for peer in peers:
                    if len(owners)>=minimum_clients:break
                    if peer not in owners and peer in metadata and negative(peer)>=minimum_count:
                        owners.append(peer);owners.sort()
                if len(owners)<minimum_clients:continue
                return dict(version=VERSION,status='METADATA_FEASIBLE',receiver=receiver,
                    experts=[dict(target=c,donor=d) for c,d in zip(classes,donors)],owners=owners,
                    selection_rule='receiver ID, class pair IDs, donor tuple IDs, filler IDs ascending after eligibility',
                    owner_CAL_negative_counts={o:negative(o) for o in owners},
                    current_peers=peers,candidate_rows=entries,
                    scope='Opportunity at current graph only; not functional-gap proof or predicted acceptance')
    return dict(version=VERSION,status='NO_FEASIBLE_TWO_CLASS_FIXTURE',candidate_rows=entries,
                selection_rule='receiver/class/donor IDs after metadata eligibility; no outcome search')


def capture_prepared_graph(*,models,clients,detectors,active_ids,task,config,ref_data,ref_labels):
    """Use the native capsule/clustering engine at the pre-optimizer boundary."""
    import torch
    from .decentralized_denice_il import _build_round_capsule
    from fed_learning.strategies.decentralized.denice_clustering import paper_context_cluster
    from .candle_v4_shadow import native_fingerprint
    before=native_fingerprint(models,detectors)
    modes={cid:[(m,m.training) for m in models[cid].modules()] for cid in active_ids}
    caps=[]
    for cid in active_ids:
        caps.append(_build_round_capsule(cid=cid,task_id=task,round_id=-1,model=models[cid],client=clients[cid],
            context_detector=detectors[cid],ref_data=ref_data[cid],ref_labels=ref_labels.get(cid),loss=0.,config=config))
        for module,mode in modes[cid]:module.training=mode
        models[cid].zero_grad(set_to_none=True)
        print(f'V4 PREFLIGHT native current-BASE descriptor client={cid} (no optimizer)',flush=True)
    result=paper_context_cluster(caps,beta=float(config.get('denice_similarity_beta',.5)),
                                 threshold=float(config['denice_similarity_threshold']))
    graph=dict(task=task,round=-1,clock='after native task allocation, before any current-task optimizer step',alpha_debug={})
    for i,cid in enumerate(active_ids):
        group=[j for j in range(len(active_ids)) if i==j or result['edges'][i,j]]
        # Strictly positive metadata weights establish eligibility only. Native aggregation
        # recomputes real contribution weights after local training each round.
        graph['alpha_debug'][cid]=dict(group_ids=[active_ids[j] for j in group],
            alphas=[1./len(group)]*len(group),similarities=[float(result['effective_similarity'][i,j]) for j in group])
    descriptors={c.client_id:dict(client=c.client_id,task=task,prototypes=c.penultimate_prototypes,
                    ternary_ages=c.ternary_ages,support=c.label_histogram,sample_count=c.sample_count) for c in caps}
    wire=sum(len(pickle.dumps(dict(sender=j,receiver=i,descriptor=descriptors[j]),protocol=5))
             for i in active_ids for j in active_ids if i!=j)
    if before!=native_fingerprint(models,detectors):raise RuntimeError('Preflight capsule capture mutated prepared native state')
    return graph,dict(application_descriptor_bytes=wire,policy='all-to-all graph descriptor projections; no model or raw samples',
        optimizer_steps=0,expert_updates=0,FIT_samples_opened=False,CAL_samples_opened=False,TEST_opened=False,
        graph_weight_policy='positive uniform eligibility weights at pre-optimizer clock, not aggregation alpha')
