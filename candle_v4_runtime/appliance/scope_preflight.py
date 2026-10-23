"""Fail-closed planning for observable certificate domains.

This module does not install patches or issue CAL/population certificates.
An input predicate cannot prove a semantic class inventory. A source contract
must be supplied by the application, registered before calibration, and backed
by an external authority. Model scores, task labels and BASE moments are not
such an authority. Checksums bind declarations; they do not authenticate them.
"""
import copy
import hashlib
import json

import numpy as np

from .config import Rejected

VERSION = 'appliance_observable_scope_preflight_v1'
FORBIDDEN_BASES = ('y_true', 'task_id', 'router_prediction', 'signature_match',
                   'head_margin', 'BASE_summary', 'training_task')


def declaration_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def checked_classes(values):
    if (not isinstance(values, list) or not values or
            any(type(c) is not int or not 0 <= c < 34 for c in values) or
            len(set(values)) != len(values)):
        raise Rejected('OBSERVABLE_SCOPE_CLASS_INVENTORY_INVALID')
    return sorted(values)


def checked_source_contract(value, registered_sha256):
    """Registration is an application trust boundary, not learned eligibility.

    The caller must establish the external source inventory independently.
    Merely writing a declaration/hash cannot provide that semantic evidence.
    This helper intentionally has no router/sketch/classifier-score arguments.
    """
    fields = {'version', 'source_id', 'domain_id', 'input_shape',
              'preprocessing_sha256', 'possible_classes', 'inventory_basis',
              'authority_reference', 'locked_before_CAL_selection'}
    if not isinstance(value, dict) or set(value) != fields:
        raise Rejected('OBSERVABLE_SCOPE_SOURCE_SCHEMA')
    if (value['version'] != VERSION or
            any(type(value[k]) is not str or not value[k]
                for k in ('source_id', 'domain_id', 'authority_reference')) or
            value['inventory_basis'] != 'external_application_source_contract' or
            value['locked_before_CAL_selection'] is not True):
        raise Rejected('OBSERVABLE_SCOPE_SOURCE_AUTHORITY_REQUIRED')
    pp = value['preprocessing_sha256']
    shape = value['input_shape']
    if (type(pp) is not str or len(pp) != 64 or any(c not in '0123456789abcdef' for c in pp) or
            not isinstance(shape, list) or not shape or
            any(type(n) is not int or n < 1 for n in shape)):
        raise Rejected('OBSERVABLE_SCOPE_INPUT_CONTRACT_INVALID')
    checked_classes(value['possible_classes'])
    if declaration_sha(value) != registered_sha256:
        raise Rejected('OBSERVABLE_SCOPE_SOURCE_NOT_PREREGISTERED')
    return copy.deepcopy(value)


def scope_plan(application_classes, observed_CAL_classes, source_contract=None,
               registered_sha256=None):
    """Separate inventory evidence from the predicate evaluable at inference.

    Returns no installation permission. CAL acceptance, function binding,
    protection conflicts and lifecycle checks remain separate mandatory gates.
    With no trusted source contract, the application-wide inventory is the only
    permissible assumption. A guessed task never narrows that inventory.
    """
    application = checked_classes(application_classes)
    observed = checked_classes(observed_CAL_classes)
    if not set(observed).issubset(application):
        raise Rejected('OBSERVABLE_SCOPE_CAL_OUTSIDE_APPLICATION_INVENTORY')
    blockers = []
    source = None
    if source_contract is None:
        inventory = application
    else:
        source = checked_source_contract(source_contract, registered_sha256)
        inventory = checked_classes(source['possible_classes'])
        if not set(inventory).issubset(application):
            raise Rejected('OBSERVABLE_SCOPE_SOURCE_OUTSIDE_APPLICATION_INVENTORY')
    missing = sorted(set(inventory) - set(observed))
    if missing:
        if source is None:
            blockers.append('NO_INDEPENDENT_OBSERVABLE_SOURCE_AUTHORITY')
        blockers.append('POSSIBLE_INPUT_CLASSES_OUTSIDE_CAL_EVIDENCE')
    return dict(version=VERSION, application_classes=application,
        observed_CAL_classes=observed, source_possible_classes=inventory,
        missing_CAL_classes=missing, source_contract=source,
        source_contract_sha256=declaration_sha(source) if source else None,
        source_authority_present=source is not None,
        domain_narrowing_requested=source is not None,
        whole_application_CAL_coverage=set(application).issubset(observed),
        evidence_coverage_feasible=not blockers, blockers=blockers,
        source_semantic_inventory_verified_by_code=False,
        install_authorized=False, population_FAR_claim=False,
        summary_is_CAL=False, per_owner_class_minimum_imposed=False,
        forbidden_membership_bases=list(FORBIDDEN_BASES))


def input_membership(x, runtime_metadata, source_contract, registered_sha256):
    """Input/metadata-only membership in a preregistered external source.

    No labels, true task ID, class inventory or model outputs are accepted in
    runtime metadata. Does not authorize a route: use only after scope_plan,
    current CAL acceptance and unchanged-function lifecycle gates pass.
    Missing/extra metadata and nonfinite inputs fail closed.
    """
    array = np.asarray(x)
    if array.ndim < 1:
        raise Rejected('OBSERVABLE_SCOPE_INPUT_BATCH_REQUIRED')
    empty = np.zeros(len(array), dtype=bool)
    source = checked_source_contract(source_contract, registered_sha256)
    if (not isinstance(runtime_metadata, dict) or
            set(runtime_metadata) != {'source_id', 'preprocessing_sha256'} or
            runtime_metadata['source_id'] != source['source_id'] or
            runtime_metadata['preprocessing_sha256'] != source['preprocessing_sha256'] or
            list(array.shape[1:]) != source['input_shape'] or
            array.dtype.kind not in 'fiu'):
        return empty
    return np.isfinite(array.reshape(len(array), int(np.prod(source['input_shape'])))).all(axis=1)
