"""Current live client endpoints; scheduler receives offers/requests only."""
import numpy as np
from .config import Rejected
from .current_calibration_data import CurrentCalibrationData
from .current_discovery import local_offers,local_requests,select_current_pairs
from .active_pair_calibration import split_counts
from .evaluate import Predictor
from .imported_route import ROUTE_RULES,stratified_roles
from .state import complete_hash,rng_snapshot,restore_rng
from .discovery_preflight import schedule_requests


class LiveCurrentClient:
    def __init__(self,client,model,router,view,seen,batch_size=512):
        if not isinstance(view,CurrentCalibrationData) or view.client_id!=client:
            raise Rejected('LIVE_DISCOVERY_CURRENT_LOCAL_AUTHORITY_REQUIRED')
        self.client=client;self.model=model;self.router=router;self.view=view
        self.seen=list(seen);self.batch_size=batch_size

    def messages(self):
        """Labels remain local, and FIT is the only prediction-based role."""
        rng=rng_snapshot()
        try:
            task=self.view.task;classes=self.view.store['task_classes'][str(task)]
            counts=self.view.manifest['clients'][str(self.client)]['role_class_counts']
            base={str(c):int(counts['base'].get(str(c),0)) for c in classes}
            cal=self.view.store['clients'][str(self.client)][str(task)]['class_counts']
            ranks=self.model.unit_ranks['fc2']
            refreshed=getattr(self.router,'router_last_refresh_task',None)
            requests=local_requests(self.client,task,classes,base,cal,ranks,refreshed)
            if refreshed!=task or not any(base[str(c)] and int(ranks[c])>=2 and
                split_counts(cal[str(c)])[0]>=32 for c in classes):
                return requests,[],[]
            pool=self.view.current_pool(self.client,'calibration',classes)
            fit=stratified_roles(pool,ROUTE_RULES['seed']+self.client)['fit']
            device=str(next(self.model.parameters()).device)
            predictions=Predictor(self.seen,device,self.batch_size)(self.model,self.router,pool['X'][fit])
            offers,failed=local_offers(self.client,task,classes,pool['y'][fit],predictions,base,ranks,
                complete_hash(self.model,self.router),self.view.store['role_manifest_sha256'])
            offers=[o for o in offers if (lambda n: n[1]>=8 and n[2]>=32)(split_counts(cal[str(o['class_id'])]))]
            return requests,offers,failed
        finally:restore_rng(rng)


def discover_live_pairs(endpoints,groups,alphas,task,round_id,preflight=None,
                        max_transactions=16,max_receivers_per_class=8):
    """Use legal positive peer lists and the full recorded alpha map.

    This is the same contract as selector.recorded_graph: peer lists exclude
    self/zero-weight neighbors; alpha maps retain self and zero-weight entries.
    """
    if set(endpoints)!=set(groups) or set(alphas)!=set(groups):
        raise Rejected('LIVE_DISCOVERY_ENTIRE_ACTIVE_GRAPH_REQUIRED')
    requests=[];offers=[];failed=[]
    for cid in sorted(groups):
        endpoint=endpoints[cid]
        donors=groups[cid];weights=alphas[cid]
        positive={d for d,v in weights.items() if d!=cid and v>0}
        if (endpoint.client!=cid or endpoint.view.task!=task or cid not in weights or
            len(set(donors))!=len(donors) or set(donors)!=positive or
            not set(weights).issubset(endpoints) or
            any(not np.isfinite(v) or v<0 for v in weights.values())):
            raise Rejected('LIVE_DISCOVERY_GRAPH_OR_TASK_CHANGED',f'client={cid}')
        r,o,f=endpoint.messages();requests.extend(r);offers.extend(o);failed.extend(f)
    rejected_requests=[]
    if preflight is None:
        selected,coverage=select_current_pairs(requests,offers,groups,alphas,task,round_id)
    else:
        if set(preflight)!=set(endpoints):raise Rejected('DISCOVERY_PREFLIGHT_ACTIVE_GRAPH_CHANGED')
        selected,coverage,rejected_requests=schedule_requests(requests,offers,groups,alphas,
            task,round_id,preflight,max_transactions,max_receivers_per_class)
    return dict(pairs=selected,coverage=coverage,requests=requests,offers=offers,rejected_offers=failed,
                task=task,round=round_id,failed_pair_substitution_allowed=False,
                selection_opened=False,holdout_predictions_used=False,
                historical_raw_CAL_reads=0,final_test_opened=False,
                receiver_preflight=preflight,rejected_requests=rejected_requests)
