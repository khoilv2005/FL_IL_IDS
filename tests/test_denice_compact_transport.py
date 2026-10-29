"""Transport equivalence, permission integrity and actual encoded byte savings."""
from dataclasses import replace
import json
import unittest

import numpy as np

from test_denice_v2_p1 import plan, contribution
from fed_learning.strategies.decentralized.denice_v2.aggregation import aggregate_round
from fed_learning.strategies.decentralized.denice_v2.compact_transport import (
    pack_contribution, unpack_contribution, transport_contributions, _serialize)
from fed_learning.strategies.decentralized.denice_v2.contracts import decode, state_hash


class CompactTests(unittest.TestCase):
    def test_zero_update_keeps_denominator_and_frozen_parameters(self):
        p = plan()
        packets = [contribution(p, 0, {'fc2.bias': [0, 0, 0, 0]}),
                   contribution(p, 1, {'fc2.bias': [0, 0, 8, 0]})]
        restored, sizes = transport_contributions(p, packets)
        dense = aggregate_round(p, packets)
        compact = aggregate_round(p, restored)
        self.assertEqual(float(compact.state.model_state['fc2.bias'][2]), 6.)
        self.assertEqual(dense.state.fingerprint(), compact.state.fingerprint())
        for a, b in zip(packets, restored):
            self.assertEqual(state_hash(a.deltas), state_hash(b.deltas))
            self.assertEqual(state_hash(a.masks), state_hash(b.masks))
        self.assertTrue(all(n > 0 for n in sizes.values()))

    def test_stale_round_and_unauthorized_update_rejected(self):
        p = plan()
        packet = contribution(p, 0, {})
        with self.assertRaisesRegex(ValueError, 'Stale'):
            pack_contribution(p, replace(packet, plan_hash='stale'))
        wire = pack_contribution(p, packet)
        p2 = replace(p, hash='another-plan')
        with self.assertRaisesRegex(ValueError, 'Stale'):
            unpack_contribution(p2, wire)
        packet.deltas['fc2.bias'][0] = 1.
        with self.assertRaisesRegex(ValueError, 'permissions'):
            pack_contribution(p, packet)

    def test_encoded_savings_on_protected_tensor_and_corrupt_mask(self):
        # Expand an existing protected layer without altering the plan's mutable schema.
        from test_denice_v2_p1 import canonical
        from fed_learning.strategies.decentralized.denice_v2.contracts import ClusterState
        from fed_learning.strategies.decentralized.denice_v2.state import snapshot
        import torch
        state = snapshot(canonical())
        state['model_state']['fc1.weight'] = torch.ones(2, 4096)
        state['algorithm_state']['weight_masks']['fc1'] = torch.ones(2, 4096)
        state['protected_parameters']['fc1.weight'] = torch.ones(2, 4096, dtype=torch.bool)
        s = ClusterState.create(**{k:v for k,v in state.items() if k not in ('model_spec','routing_space_hash')})
        p = plan(s)
        packet = contribution(p, 0, {})
        message = pack_contribution(p, packet)
        from dataclasses import asdict
        dense_bytes = len(_serialize(dict(codec='dense_v1',packet=asdict(packet))))
        self.assertLess(len(message), dense_bytes / 4)
        payload = decode(json.loads(message))
        self.assertEqual(payload['codec'], 'masked_v1')
        unpack_contribution(p, message)
        payload['tensors']['fc1.weight']['mask'] = np.array([255], np.uint8)
        with self.assertRaisesRegex(ValueError, 'shape/mask'):
            unpack_contribution(p, _serialize(payload))


if __name__ == '__main__':
    unittest.main()
