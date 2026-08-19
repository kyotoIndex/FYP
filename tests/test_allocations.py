import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedhello_sst2 import (
    ClientSpec,
    LoRALinear,
    MethodPlan,
    allocate_ranks_by_importance,
    build_client_rows,
    client_positive_ratios,
    component_round_schedule,
    component_communication_parameter_counts,
    constrained_dirichlet_positive_counts,
    constrained_dirichlet_label_counts,
    controlled_capability_counts,
    deterministic_geometric_layers,
    fim_allocation_probabilities,
    geometric_prior,
    formal_active_layers,
    legend_arithmetic_ranks,
    legend_depths,
    legend_simulated_completion,
    is_online_layer_method,
    ema_update,
    online_layer_scores,
    random_component_allocation,
    read_text_rows,
    resolve_online_layer_config,
    sample_without_replacement,
    select_online_layers,
    select_component_allocation,
    should_refresh_client_importance,
    should_refresh_fim,
)


class AllocationTests(unittest.TestCase):
    def setUp(self):
        capacities = [2, 2, 2, 3, 4]
        self.clients = [
            ClientSpec(index, "test", capacity, 1.0, 1.0, 0.5, 10)
            for index, capacity in enumerate(capacities)
        ]

    def test_five_client_ratio_keeps_all_resource_levels(self):
        self.assertEqual(controlled_capability_counts([0.6, 0.3, 0.1], 5), [3, 1, 1])

    def test_geometric_masks_have_exact_capacity(self):
        for pattern in ("triangle", "inverted_triangle", "bottleneck", "uniform"):
            for capacity in (2, 3, 4):
                layers = deterministic_geometric_layers(pattern, capacity, 4)
                self.assertEqual(len(layers), capacity)
                self.assertEqual(len(set(layers)), capacity)

    def test_bottleneck_prior_prefers_edges(self):
        prior = geometric_prior("bottleneck", self.clients, 4)
        self.assertAlmostEqual(sum(prior), 1.0)
        self.assertGreater(prior[0], prior[2])
        self.assertGreater(prior[3], prior[2])

    def test_fim_probabilities_are_normalized_and_ranked(self):
        probabilities = fim_allocation_probabilities([1.0, 2.0, 10.0, 3.0], self.clients)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(max(range(4), key=probabilities.__getitem__), 2)

    def test_weighted_sampling_is_reproducible(self):
        first = sample_without_replacement([0.1, 0.2, 0.3, 0.4], 2, torch.Generator().manual_seed(7))
        second = sample_without_replacement([0.1, 0.2, 0.3, 0.4], 2, torch.Generator().manual_seed(7))
        self.assertEqual(first, second)

    def test_periodic_fim_refresh_starts_after_rgd_without_double_refresh(self):
        refreshed = []
        cached = False
        for round_id in range(1, 202):
            refresh = should_refresh_fim("fedhello", round_id, 50, 50, cached)
            if refresh:
                refreshed.append(round_id)
                cached = True
        self.assertEqual(refreshed, [51, 101, 151, 201])

    def test_fim_hla_refreshes_every_round(self):
        self.assertTrue(should_refresh_fim("fim_hla", 1, 0, 1, False))
        self.assertTrue(should_refresh_fim("fim_hla", 2, 0, 1, True))

    def test_every_configured_allocation_method_has_exact_capacity(self):
        methods = (
            "fedlora",
            "resource_only",
            "fedra_random_partial",
            "fim_hla",
            "gd_hla_triangle",
            "gd_hla_inverted_triangle",
            "gd_hla_bottleneck",
            "gd_hla_uniform",
            "rgd_hla",
            "fedhello",
        )
        from fedhello_sst2 import allocate_layers

        for method in methods:
            for client in self.clients:
                layers, _, _ = allocate_layers(
                    method,
                    client,
                    4,
                    torch.Generator().manual_seed(7),
                    [0.1, 0.2, 0.3, 0.4],
                    [0.3, 0.2, 0.2, 0.3],
                    2,
                    1,
                )
                expected = 4 if method == "fedlora" else client.capacity
                self.assertEqual(len(layers), expected, method)
                self.assertEqual(len(set(layers)), expected, method)

    def test_legend_paper_rank_plan_is_arithmetic_and_within_budget(self):
        ranks = legend_arithmetic_ranks(12, 192, common_difference=1)
        self.assertEqual(ranks, list(range(10, 22)))
        self.assertLessEqual(sum(ranks), 192)
        self.assertEqual({right - left for left, right in zip(ranks, ranks[1:])}, {1})

    def test_legend_assigns_more_depth_to_faster_clients_and_obeys_budget(self):
        ranks = legend_arithmetic_ranks(12, 192, common_difference=1)
        depths = legend_depths(
            {0: 1.0, 1: 2.0, 2: 4.0},
            12,
            ranks,
            {0: 192, 1: 144, 2: 96},
        )
        self.assertGreaterEqual(depths[0], depths[1])
        self.assertGreaterEqual(depths[1], depths[2])
        for client_id, budget in {0: 192, 1: 144, 2: 96}.items():
            self.assertLessEqual(sum(ranks[12 - depths[client_id] :]), budget)

    def test_legend_simulated_completion_is_deterministic_and_resource_aware(self):
        ranks = list(range(10, 22))
        slow = ClientSpec(0, "low", 6, 1.8, 1.3, 0.5, 512)
        fast = ClientSpec(1, "high", 12, 0.7, 0.8, 0.5, 512)
        active = list(range(7, 12))
        slow_time = legend_simulated_completion(slow, active, ranks, 16)
        fast_time = legend_simulated_completion(fast, active, ranks, 16)
        self.assertGreater(slow_time, fast_time)
        self.assertEqual(
            slow_time,
            legend_simulated_completion(slow, active, ranks, 16),
        )

    def test_importance_ranks_keep_budget_and_bounds(self):
        importance = [1, 1, 2, 3, 5, 8, 13, 21, 13, 8, 5, 3]
        ranks = allocate_ranks_by_importance(importance, 192, 4, 32)
        self.assertEqual(sum(ranks), 192)
        self.assertTrue(all(4 <= rank <= 32 for rank in ranks))
        self.assertGreater(ranks[7], ranks[0])

    def test_formal_layer_policies_have_expected_capacity(self):
        client = self.clients[0]
        importance = [0.01, 0.02, 0.03, 0.84, 0.04, 0.06]
        ranks = [4, 8, 12, 32, 20, 20]
        plan = MethodPlan(ranks=ranks, global_importance=importance)
        layers, _ = formal_active_layers("importance_resource_lora", client, plan, 6)
        self.assertEqual(len(layers), client.capacity)
        self.assertEqual(len(set(layers)), client.capacity)

        personalized = MethodPlan(
            ranks=ranks,
            client_clusters={client.client_id: 1},
            cluster_importance={1: importance},
        )
        first, _ = formal_active_layers("personalized_cost_aware_lora", client, personalized, 6)
        second, _ = formal_active_layers("personalized_cost_aware_lora", client, personalized, 6)
        self.assertEqual(first, second)
        self.assertEqual(len(first), client.capacity)

    def test_dirichlet_partition_ratios_are_seeded_and_non_iid(self):
        config = {"data_heterogeneity": {"mode": "dirichlet_label_skew", "alpha": 0.3}}
        first = client_positive_ratios(config, 20, 0.5, 7)
        second = client_positive_ratios(config, 20, 0.5, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertGreater(max(first) - min(first), 0.5)

    def test_constrained_dirichlet_preserves_global_label_count(self):
        first = constrained_dirichlet_positive_counts(
            total_clients=5,
            samples_per_client=512,
            global_positive_ratio=0.5578256544,
            alpha=0.3,
            seed=21,
        )
        second = constrained_dirichlet_positive_counts(
            total_clients=5,
            samples_per_client=512,
            global_positive_ratio=0.5578256544,
            alpha=0.3,
            seed=21,
        )
        self.assertEqual(first, second)
        self.assertEqual(sum(first), round(5 * 512 * 0.5578256544))
        self.assertTrue(all(0 <= count <= 512 for count in first))
        self.assertGreater(max(first) - min(first), 100)

    def test_multiclass_constrained_dirichlet_preserves_all_margins(self):
        ratios = [0.4, 0.3, 0.2, 0.1]
        counts = constrained_dirichlet_label_counts(
            total_clients=20,
            samples_per_client=512,
            global_label_ratios=ratios,
            alpha=0.3,
            seed=7,
        )
        self.assertEqual(len(counts), 20)
        self.assertTrue(all(sum(row) == 512 for row in counts))
        expected = [4096, 3072, 2048, 1024]
        actual = [
            sum(row[label] for row in counts)
            for label in range(4)
        ]
        self.assertEqual(actual, expected)
        self.assertGreater(
            max(row[0] for row in counts) - min(row[0] for row in counts),
            100,
        )

    def test_named_tsv_reads_sentence_pairs_and_label_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.tsv"
            path.write_text(
                "index\tquestion\tsentence\tlabel\n"
                "0\tWho?\tA person.\tentailment\n"
                "1\tWhere?\tNowhere.\tnot_entailment\n",
                encoding="utf-8",
            )
            rows = read_text_rows(
                path,
                {
                    "format": "named_tsv",
                    "text_columns": ["question", "sentence"],
                    "label_column": "label",
                    "label_mapping": {
                        "entailment": 0,
                        "not_entailment": 1,
                    },
                },
            )
        self.assertEqual(
            rows,
            [
                ("Who?", "A person.", 0),
                ("Where?", "Nowhere.", 1),
            ],
        )

    def test_component_communication_counts_explicit_sync(self):
        query = LoRALinear(nn.Linear(4, 4), 4, 4.0, 0.0)
        value = LoRALinear(nn.Linear(4, 4), 4, 4.0, 0.0)
        layer = SimpleNamespace(
            attention=SimpleNamespace(
                self=SimpleNamespace(query=query, value=value)
            )
        )
        model = SimpleNamespace(
            bert=SimpleNamespace(
                encoder=SimpleNamespace(layer=[layer])
            ),
            classifier=nn.Linear(4, 2),
        )
        allocation = {0: [0, 1]}
        no_sync = component_communication_parameter_counts(
            model, allocation, "a", False
        )
        full_sync = component_communication_parameter_counts(
            model, allocation, "a", True
        )
        self.assertEqual(no_sync, (42, 26, 0))
        self.assertEqual(full_sync, (74, 26, 64))

    def test_constrained_dirichlet_ratios_use_fixed_client_size(self):
        config = {
            "data_heterogeneity": {
                "mode": "constrained_dirichlet_label_skew",
                "alpha": 0.3,
            }
        }
        ratios = client_positive_ratios(
            config,
            total_clients=5,
            global_positive_ratio=0.5578256544,
            seed=7,
            samples_per_client=512,
        )
        counts = [round(512 * ratio) for ratio in ratios]
        self.assertEqual(sum(counts), round(5 * 512 * 0.5578256544))
        self.assertGreater(max(ratios) - min(ratios), 0.2)

    def test_client_partition_samples_without_replacement(self):
        clients = [
            ClientSpec(0, "low", 2, 1.0, 1.0, 0.5, 4),
            ClientSpec(1, "high", 4, 1.0, 1.0, 0.5, 4),
        ]
        source = [
            (f"positive-{index}", 1) for index in range(10)
        ] + [
            (f"negative-{index}", 0) for index in range(10)
        ]
        rows = build_client_rows(source, clients, seed=7)
        assigned = [row for client_rows in rows.values() for row in client_rows]
        self.assertEqual(len(assigned), 8)
        self.assertEqual(len(set(assigned)), 8)

    def test_client_partition_rejects_class_reuse(self):
        clients = [
            ClientSpec(0, "low", 2, 1.0, 1.0, 1.0, 4),
            ClientSpec(1, "high", 4, 1.0, 1.0, 1.0, 4),
        ]
        source = [(f"positive-{index}", 1) for index in range(6)]
        with self.assertRaisesRegex(ValueError, "without reuse"):
            build_client_rows(source, clients, seed=7)

    def test_online_ema_is_normalized_and_uses_history(self):
        first = ema_update(None, [1.0, 3.0], 0.8)
        second = ema_update(first, [3.0, 1.0], 0.8)
        self.assertAlmostEqual(sum(second), 1.0)
        self.assertGreater(second[1], second[0])
        self.assertGreater(second[0], first[0])

    def test_online_exploration_rewards_under_selected_layers(self):
        scores = online_layer_scores(
            importance=[1.0, 1.0, 1.0],
            selection_counts=[10, 0, 10],
            last_selected_rounds=[4, -1, 4],
            previous_layers=[],
            round_id=5,
            exploration_beta=0.5,
            freshness_gamma=0.1,
            stability_delta=0.0,
        )
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], scores[2])

    def test_online_selection_has_capacity_and_stability_bonus(self):
        layers, scores = select_online_layers(
            importance=[1.0, 1.0, 1.0, 1.0],
            selection_counts=[1, 1, 1, 1],
            last_selected_rounds=[1, 1, 1, 1],
            previous_layers=[2, 3],
            capacity=2,
            round_id=2,
            exploration_beta=0.0,
            freshness_gamma=0.0,
            stability_delta=0.2,
        )
        self.assertEqual(layers, [2, 3])
        self.assertEqual(len(scores), 4)

    def test_online_variant_overrides_only_named_parameters(self):
        config = {
            "pcla_online_layer": {
                "importance_ema": 0.8,
                "exploration_beta": 0.15,
                "stability_delta": 0.05,
            },
            "pcla_online_layer_variants": {
                "pcla_online_layer_b100_s000": {
                    "exploration_beta": 1.0,
                    "stability_delta": 0.0,
                }
            },
        }
        resolved = resolve_online_layer_config(
            "pcla_online_layer_b100_s000", config
        )
        self.assertTrue(is_online_layer_method("pcla_online_layer_b100_s000"))
        self.assertEqual(resolved["importance_ema"], 0.8)
        self.assertEqual(resolved["exploration_beta"], 1.0)
        self.assertEqual(resolved["stability_delta"], 0.0)

    def test_component_allocation_respects_layer_and_component_budgets(self):
        importance = [
            [float(layer * 4 + component + 1) for component in range(4)]
            for layer in range(3)
        ]
        allocation, scores = select_component_allocation(
            importance=importance,
            selection_counts=[0] * 12,
            last_selected_rounds=[-1] * 12,
            previous_components=[],
            layer_capacity=2,
            components_per_layer=2,
            round_id=1,
            exploration_beta=0.15,
            freshness_gamma=0.05,
            stability_delta=0.0,
        )
        self.assertEqual(len(allocation), 2)
        self.assertEqual(sum(map(len, allocation.values())), 4)
        self.assertTrue(all(allocation.values()))
        self.assertEqual(len(scores), 12)

    def test_random_component_allocation_is_seeded_and_budget_matched(self):
        first = random_component_allocation(
            4, 8, 2, 3, torch.Generator().manual_seed(7)
        )
        second = random_component_allocation(
            4, 8, 2, 3, torch.Generator().manual_seed(7)
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(sum(map(len, first.values())), 6)

    def test_component_warm_start_schedule_resets_alternating_factor(self):
        config = {"pcla_component": {"warm_start_rounds": 2}}
        self.assertEqual(
            component_round_schedule(
                "pcla_component_warmstart", 1, config
            ),
            (True, "both", 0),
        )
        self.assertEqual(
            component_round_schedule(
                "pcla_component_warmstart", 2, config
            ),
            (True, "both", 0),
        )
        self.assertEqual(
            component_round_schedule(
                "pcla_component_warmstart", 3, config
            ),
            (False, "b", 1),
        )
        self.assertEqual(
            component_round_schedule(
                "pcla_component_warmstart", 4, config
            ),
            (False, "a", 2),
        )

    def test_original_component_schedule_has_no_warm_start(self):
        config = {"pcla_component": {"warm_start_rounds": 2}}
        self.assertEqual(
            component_round_schedule(
                "pcla_component_alternating", 1, config
            ),
            (False, "b", 1),
        )

    def test_client_importance_refresh_initializes_late_participant(self):
        self.assertTrue(should_refresh_client_importance(None, 2, 3))
        self.assertFalse(should_refresh_client_importance(2, 3, 3))
        self.assertFalse(should_refresh_client_importance(2, 4, 3))
        self.assertTrue(should_refresh_client_importance(2, 5, 3))


if __name__ == "__main__":
    unittest.main()
