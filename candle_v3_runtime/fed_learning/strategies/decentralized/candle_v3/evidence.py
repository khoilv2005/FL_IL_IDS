"""Reject impossible current-evidence budgets before model transmission or fitting."""


def current_evidence_preflight(data, receiver, donors, targets, task, settings, ledger):
    bounds = {}
    if not hasattr(data,'count_upper_bounds'):
        return dict(possible=False,reason='Current evidence metadata unavailable',bounds={})
    for owner in [receiver]+list(donors):
        row = {role:data.count_upper_bounds(owner,role,task=task,per_class=cap)
               for role,cap in (('fit',settings.fit_per_class),('calibration',settings.cal_per_class))}
        if owner != receiver:
            row = ledger.transmit(owner,receiver,'verification_receipt',
                dict(role='current_evidence_upper_bounds',task=task,counts=row))['counts']
        bounds[owner] = row
    missing = []
    for c in targets:
        fit = sum(bounds[o]['fit'].get(c,0) for o in donors)
        cal = sum(bounds[o]['calibration'].get(c,0) for o in bounds)
        if fit < settings.min_fit_positive:
            missing.append(f'class={c}: FIT upper bound {fit} < {settings.min_fit_positive}')
        if cal < settings.min_cal_positive:
            missing.append(f'class={c}: CAL upper bound {cal} < {settings.min_cal_positive}')
    negatives = sum(n for c,n in bounds[receiver]['calibration'].items() if c not in targets)
    if negatives < settings.min_cal_negative:
        missing.append(f'receiver negative CAL upper bound {negatives} < {settings.min_cal_negative}')
    return dict(possible=not missing,reason='; '.join(missing) if missing else 'Raw counts are possible; unique acceptance still required',
                bounds=bounds,missing_evidence=missing,
                authority='locked current-task role counts only; no CAL scores or raw examples')
