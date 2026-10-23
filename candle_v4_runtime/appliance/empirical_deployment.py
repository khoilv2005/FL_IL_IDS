"""Explicit empirical deployment permission, distinct from CAL evidence scope.

This declaration permits an initially accepted unchanged guard on all application
inputs. It does not certify unseen classes or claim population safety. Function
drift and observed protection conflicts still suspend the imported route.
"""
import copy

from .config import Rejected
from .cumulative_certificate import sha

VERSION = 'appliance_empirical_current_CAL_v2'


def deployment_scope(entry):
    declaration = entry.get('empirical_deployment')
    if declaration is None:
        return None
    fields = {'version', 'initial_certificate_sha256', 'patch_id', 'receiver',
              'guard_function_fingerprint', 'scope', 'evidence_scope_is_not_deployment_scope',
              'population_FAR_claim', 'threshold_changes_allowed',
              'observed_conflict_suspends', 'function_drift_suspends', 'guard_backend', 'torch_version'}
    if (not isinstance(declaration, dict) or set(declaration) != fields or
            entry.get('empirical_deployment_sha256') != sha(declaration) or
            declaration['version'] != VERSION or
            any(declaration[k] != entry[k] for k in ('patch_id', 'receiver', 'guard_function_fingerprint')) or
            declaration['initial_certificate_sha256'] != entry['lifecycle_certificate_sha256'] or
            declaration['evidence_scope_is_not_deployment_scope'] is not True or
            declaration['population_FAR_claim'] is not False or
            declaration['threshold_changes_allowed'] is not False or
            declaration['observed_conflict_suspends'] is not True or
            declaration['function_drift_suspends'] is not True or
            declaration['guard_backend'] not in ('cpu', 'cuda') or
            not isinstance(declaration['torch_version'], str) or not declaration['torch_version']):
        raise Rejected('EMPIRICAL_DEPLOYMENT_DECLARATION_CHANGED')
    scope = declaration['scope']
    if (not isinstance(scope, dict) or set(scope) != {'domain_id', 'classes'} or
            scope['domain_id'] != entry['lifecycle_certificate']['scope']['domain_id'] or
            scope['classes'] != list(range(34))):
        raise Rejected('EMPIRICAL_DEPLOYMENT_DOMAIN_CHANGED')
    return copy.deepcopy(scope)


def bind_deployment(entry, backend, torch_version):
    if 'empirical_deployment' in entry:
        return deployment_scope(entry)
    declaration = dict(version=VERSION,
        initial_certificate_sha256=entry['lifecycle_certificate_sha256'],
        patch_id=entry['patch_id'], receiver=entry['receiver'],
        guard_function_fingerprint=entry['guard_function_fingerprint'],
        scope=dict(domain_id=entry['lifecycle_certificate']['scope']['domain_id'], classes=list(range(34))),
        guard_backend=backend, torch_version=torch_version,
        evidence_scope_is_not_deployment_scope=True, population_FAR_claim=False,
        threshold_changes_allowed=False, observed_conflict_suspends=True, function_drift_suspends=True)
    entry.update(empirical_deployment=declaration, empirical_deployment_sha256=sha(declaration))
    return deployment_scope(entry)
