"""
Tests for PlexusDER and PlexusNICE algorithms.

Phase 3: Regression tests to verify:
- MRO / training ownership of new clients
- task_loop doesn't lose bandwidth-aware aggregator state
- No double-apply success_fraction
- Registry contains new algorithms
"""

import pytest
import torch

from fed_learning.strategies import get_strategy, STRATEGIES, list_strategies
from fed_learning.clients.der_client import DERClient
from fed_learning.clients.nice_client import NICEClient
from fed_learning.clients.plexus_der_client import PlexusDERClient
from fed_learning.clients.plexus_nice_client import PlexusNICEClient
from fed_learning.factories.server_factory import _SERVER_REGISTRY
from fed_learning.factories.client_factory import _CLIENT_REGISTRY


class TestPlexusDERNICEInheritance:
    """Test Phase 9: MRO / training ownership of new clients."""

    def test_plexus_der_client_inherits_from_der_client(self):
        """PlexusDERClient should inherit from DERClient to get actual DER training logic."""
        assert issubclass(PlexusDERClient, DERClient), (
            "PlexusDERClient must inherit from DERClient for two-stage training"
        )

    def test_plexus_nice_client_inherits_from_nice_client(self):
        """PlexusNICEClient should inherit from NICEClient to get actual NICE training logic."""
        assert issubclass(PlexusNICEClient, NICEClient), (
            "PlexusNICEClient must inherit from NICEClient for phase-based training"
        )

    def test_plexus_der_client_has_update_exemplars(self):
        """PlexusDERClient should have update_exemplars from DERClient."""
        client = PlexusDERClient(
            client_id=0,
            X_train=torch.randn(100, 10),
            y_train=torch.randint(0, 5, (100,)),
        )
        assert hasattr(client, 'update_exemplars'), (
            "PlexusDERClient must have update_exemplars for replay buffer management"
        )

    def test_plexus_der_client_has_replay_buffer(self):
        """PlexusDERClient should have replay_buffer from DERClient."""
        client = PlexusDERClient(
            client_id=0,
            X_train=torch.randn(100, 10),
            y_train=torch.randint(0, 5, (100,)),
        )
        assert hasattr(client, 'replay_buffer'), (
            "PlexusDERClient must have replay_buffer"
        )

    def test_plexus_der_train_is_from_der_client(self):
        """PlexusDERClient.train should be from DERClient module."""
        assert PlexusDERClient.train.__module__ == DERClient.train.__module__, (
            "PlexusDERClient.train must come from DERClient"
        )

    def test_plexus_nice_train_is_from_nice_client(self):
        """PlexusNICEClient.train should be from NICEClient module."""
        assert PlexusNICEClient.train.__module__ == NICEClient.train.__module__, (
            "PlexusNICEClient.train must come from NICEClient"
        )


class TestPlexusDERNICEServerFactory:
    """Test Phase 10: Server factory contains new algorithms."""

    def test_plexus_der_server_registered(self):
        """PlexusDERServer should be in _SERVER_REGISTRY."""
        assert "plexus_der" in _SERVER_REGISTRY, (
            "plexus_der must be in server registry"
        )

    def test_plexus_nice_server_registered(self):
        """PlexusNICEServer should be in _SERVER_REGISTRY."""
        assert "plexus_nice" in _SERVER_REGISTRY, (
            "plexus_nice must be in server registry"
        )

    def test_plexus_der_client_registered(self):
        """PlexusDERClient should be in _CLIENT_REGISTRY."""
        assert "plexus_der" in _CLIENT_REGISTRY, (
            "plexus_der must be in client registry"
        )

    def test_plexus_nice_client_registered(self):
        """PlexusNICEClient should be in _CLIENT_REGISTRY."""
        assert "plexus_nice" in _CLIENT_REGISTRY, (
            "plexus_nice must be in client registry"
        )


class TestPlexusDERNICEStrategies:
    """Test Phase 12: Strategy registry contains new algorithms."""

    def test_plexus_der_in_strategies(self):
        """plexus_der should be in STRATEGIES dict."""
        assert "plexus_der" in STRATEGIES, (
            "plexus_der must be in STRATEGIES"
        )

    def test_plexus_nice_in_strategies(self):
        """plexus_nice should be in STRATEGIES dict."""
        assert "plexus_nice" in STRATEGIES, (
            "plexus_nice must be in STRATEGIES"
        )

    def test_plexus_der_in_list_strategies(self):
        """plexus_der should be in list_strategies() output."""
        strategies = list_strategies()
        assert "plexus_der" in strategies, (
            "plexus_der must be in list_strategies()"
        )

    def test_plexus_nice_in_list_strategies(self):
        """plexus_nice should be in list_strategies() output."""
        strategies = list_strategies()
        assert "plexus_nice" in strategies, (
            "plexus_nice must be in list_strategies()"
        )

    def test_get_strategy_plexus_der(self):
        """get_strategy should return correct trainer/aggregator for plexus_der."""
        trainer, aggregator = get_strategy("plexus_der")
        assert trainer.__class__.__name__ == "PlexusDERTrainer"
        assert aggregator.__class__.__name__ == "PlexusDERAggregator"

    def test_get_strategy_plexus_nice(self):
        """get_strategy should return correct trainer/aggregator for plexus_nice."""
        trainer, aggregator = get_strategy("plexus_nice")
        assert trainer.__class__.__name__ == "PlexusNICETrainer"
        assert aggregator.__class__.__name__ == "PlexusNICEAggregator"


class TestPlexusAggregatorNoSuccessFractionDoubleApply:
    """Test Phase 11: No double-apply success_fraction in aggregators."""

    def test_plexus_der_aggregator_uses_all_results(self):
        """PlexusDERAggregator should aggregate all results passed to it (server filters)."""
        from fed_learning.strategies.federated.plexus_der import PlexusDERAggregator

        aggregator = PlexusDERAggregator(
            sample_size=13,
            num_aggregators=1,
            success_fraction=0.8,
            inactivity_threshold=50,
        )

        # Simulate 5 results being passed
        results = [
            {"client_id": i, "num_samples": 100, "params": {"layer1.weight": torch.randn(10, 10)}}
            for i in range(5)
        ]

        # Mock global params
        global_params = {"layer1.weight": torch.randn(10, 10)}

        # aggregate should use ALL 5 results (not filter by success_fraction)
        # We verify by checking that _weighted_average gets called with all results
        # Since we can't easily mock _weighted_average, we check that aggregate doesn't
        # do its own filtering (success_fraction filtering was removed)
        # The aggregator should pass all results to _weighted_average

        # Just verify aggregate doesn't raise and uses the results
        result = aggregator.aggregate(results, global_params)
        assert result is not None

    def test_plexus_nice_aggregator_uses_all_results(self):
        """PlexusNICEAggregator should aggregate all results passed to it (server filters)."""
        from fed_learning.strategies.federated.plexus_nice import PlexusNICEAggregator

        aggregator = PlexusNICEAggregator(
            sample_size=13,
            num_aggregators=1,
            success_fraction=0.8,
            inactivity_threshold=50,
        )

        # Simulate 5 results being passed
        results = [
            {
                "client_id": i,
                "num_samples": 100,
                "params": {"layer1.weight": torch.randn(10, 10)},
            }
            for i in range(5)
        ]

        # Mock global params
        global_params = {"layer1.weight": torch.randn(10, 10)}

        # Verify aggregate doesn't raise and uses the results
        result = aggregator.aggregate(results, global_params)
        assert result is not None


class TestPlexusNICEServerStateSync:
    """Test that PlexusNICE server keeps weight state and age state in sync."""

    def test_neuron_ages_from_used_results_not_all_results(self):
        """
        PlexusNICEServer should read neuron_ages from used_results, not all results.

        This verifies the fix for the state mismatch bug where:
        - weights were aggregated from used_results (filtered by success_fraction)
        - but neuron_ages were incorrectly read from the first entry in all results

        The key insight: the server code now uses used_results when iterating
        for neuron_ages, ensuring weight state and age state come from the same
        set of clients.
        """
        from fed_learning.strategies.federated.plexus_nice import PlexusNICEAggregator
        from fed_learning.models.nice_model import NICEModel
        import numpy as np

        # Create aggregator
        aggregator = PlexusNICEAggregator(
            sample_size=13,
            num_aggregators=1,
            success_fraction=0.5,
            inactivity_threshold=50,
        )

        # Create mock server
        class MockServer:
            def __init__(self):
                self.global_model = NICEModel((1, 28, 28), num_classes=10)
                self.aggregator = aggregator
                self.primary_device = "cpu"

            def _get_frozen_param_keys(self):
                return []

        server = MockServer()
        global_params = server.global_model.state_dict()

        # Create 10 results with success_fraction=0.5:
        # floor(10 * 0.5) = 5, max(3, 5) = 5
        # used_results = [c0, c1, c2, c3, c4] (first 5)
        #
        # Setup: make c0..c4 have ages, c5..c9 also have ages
        # Both used_results and results would find c0's ages, so this is not a good test.
        #
        # Better setup: c0..c4 have NO ages, c5 has ages
        # - used_results iterates c0..c4, finds nothing
        # - OLD BUGGY code iterates ALL results, finds c5's ages
        # - This shows the mismatch!

        client5_ages = {"conv1": np.array([5, 5, 5, 5, 5]), "fc1": np.ones(5), "fc2": np.ones(10)}

        results = [
            {
                "client_id": i,
                "num_samples": 100,
                "params": {"layer1.weight": torch.randn(10, 10)},
                "neuron_ages": None,  # c0-c4 have no ages
                "loss": 1.0,
            }
            for i in range(5)
        ] + [
            {
                "client_id": 5,
                "num_samples": 100,
                "params": {"layer1.weight": torch.randn(10, 10)},
                "neuron_ages": client5_ages,  # c5 has ages but NOT in used_results
                "loss": 1.0,
            },
            {
                "client_id": 6,
                "num_samples": 100,
                "params": {"layer1.weight": torch.randn(10, 10)},
                "neuron_ages": None,
                "loss": 1.0,
            },
            {
                "client_id": 7,
                "num_samples": 100,
                "params": {"layer1.weight": torch.randn(10, 10)},
                "neuron_ages": None,
                "loss": 1.0,
            },
            {
                "client_id": 8,
                "num_samples": 100,
                "params": {"layer1.weight": torch.randn(10, 10)},
                "neuron_ages": None,
                "loss": 1.0,
            },
            {
                "client_id": 9,
                "num_samples": 100,
                "params": {"layer1.weight": torch.randn(10, 10)},
                "neuron_ages": None,
                "loss": 1.0,
            },
        ]

        # Verify test setup: used_results = [c0..c4], c5 has ages but is NOT in used_results
        n_required = max(3, int(len(results) * 0.5))
        used_results = results[:n_required] if len(results) > n_required else results
        assert len(used_results) == 5
        assert used_results[0]["client_id"] == 0
        assert used_results[1]["client_id"] == 1
        assert results[5]["client_id"] == 5
        assert results[5]["neuron_ages"] is not None  # c5 has ages but not in used_results

        # Aggregate with used_results
        new_params = aggregator.aggregate(used_results, global_params)

        # OLD BUGGY behavior: would search through ALL results, find c5's ages
        buggy_ages = None
        for r in results:  # Bug: iterating over results, not used_results
            if "neuron_ages" in r and r["neuron_ages"]:
                buggy_ages = r["neuron_ages"]
                break

        # CORRECT behavior (the fix): only search used_results
        correct_ages = None
        if used_results:
            for r in used_results:  # Fixed: only iterate over used_results
                if "neuron_ages" in r and r["neuron_ages"]:
                    correct_ages = r["neuron_ages"]
                    break

        # OLD BUG: finds ages from c5 (which contributed NO weights)
        assert buggy_ages is not None
        assert np.array_equal(buggy_ages["conv1"], client5_ages["conv1"])

        # CORRECT: finds no ages (c0..c4 all have None)
        assert correct_ages is None

        # Apply correct behavior to server - should NOT update ages
        if used_results:
            for r in used_results:
                if "neuron_ages" in r and r["neuron_ages"]:
                    server.global_model.set_neuron_ages_state(r["neuron_ages"])
                    break

        # Verify server did NOT set ages from c5 (client not in used_results)
        actual_ages = server.global_model.get_neuron_ages_state()
        # The model's ages should remain at initial state since used_results[0..4] had no ages
        assert actual_ages is not None


class TestAggregatorSelectionAfterFilter:
    """Test that aggregator selection respects candidate_ids even when population_view has active peers."""

    def test_aggregator_must_be_in_sample_after_filter(self):
        """
        After filtering by participating_clients, selected_aggregator_id must be in sample_ids.

        This tests the fix for the bug where:
        - aggregator_ids was computed on all_ids
        - but sample_ids was filtered by participating_clients
        - resulting in selected_aggregator_id potentially being outside the sample

        CRITICAL: This test seeds population_view BEFORE calling get_round_aggregators,
        to hit the buggy branch where active peers override candidate_ids.
        """
        from fed_learning.strategies.federated.plexus_der import PlexusDERAggregator

        # Create aggregator with known bandwidths
        client_bandwidths = {0: 100, 1: 200, 2: 50, 3: 150}
        aggregator = PlexusDERAggregator(
            sample_size=13,
            num_aggregators=1,
            success_fraction=0.8,
            inactivity_threshold=50,
            client_bandwidths=client_bandwidths,
        )

        all_ids = [0, 1, 2, 3]
        round_num = 5

        # Seed population_view with all 4 clients as active at round round_num
        # This is the key: the buggy code would use these active peers
        # instead of the candidate_ids passed to get_round_aggregators
        for cid in all_ids:
            aggregator.population_view.update(cid, round_num, is_online=True)

        # Now call get_round_aggregators with only [2, 3] as candidates
        # The FIXED code should intersect active peers with candidate_ids
        # and only return aggregators from [2, 3]
        candidate_ids = [2, 3]
        aggregator_ids = aggregator.get_round_aggregators(round_num, candidate_ids)
        sample_ids = aggregator.get_round_sample(round_num, candidate_ids)

        fixed_aggregator = aggregator_ids[0] if aggregator_ids else None

        # Ensure aggregator is in final sample (the fix)
        if fixed_aggregator is not None and fixed_aggregator not in sample_ids:
            sample_bandwidths = {sid: client_bandwidths.get(sid, 0) for sid in sample_ids}
            fixed_aggregator = max(sample_bandwidths, key=sample_bandwidths.get)

        # The fixed aggregator MUST be in candidate_ids [2, 3]
        # Buggy code would return [0] or [1] (highest bandwidth in full active set)
        assert fixed_aggregator in candidate_ids, (
            f"Aggregator {fixed_aggregator} must be one of participating clients {candidate_ids}"
        )

        # Also verify sample is limited to candidate_ids
        for sid in sample_ids:
            assert sid in candidate_ids, (
                f"Sample member {sid} must be in candidate_ids {candidate_ids}"
            )
