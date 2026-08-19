# PCLA Experiments

This is the active and self-contained FYP experiment project. It contains the
Fed-HeLLo mechanism reproduction, comparison baselines, PCLA-v1, audited raw
results, and publication-ready plots.

## Layout

```text
pcla_experiments/
|-- configs/      Versioned experiment protocols
|-- data/         Local SST-2 TSV files used by the runner
|-- docs/         Research interpretation and provenance notes
|-- results/      Immutable completed runs and generated figures
|-- scripts/      Run, validate, aggregate, and plot entry points
|-- src/          Training and allocation implementation
|-- tests/        Allocation and scheduling unit tests
|-- README.md
`-- requirements.txt
```

All commands below start from this directory:

```powershell
cd F:\FYP\pcla_experiments
```

## Environment

The completed experiment used:

- Python 3.12.6
- PyTorch 2.5.1 + CUDA 12.1
- Transformers 4.57.1
- NumPy 2.1.2
- pandas 2.2.3
- Matplotlib 3.9.2
- NVIDIA RTX 4070 Ti SUPER

Install the Python dependencies in a clean environment. Install the PyTorch
build appropriate for the target CUDA version first, then run:

```powershell
pip install -r requirements.txt
```

The model is loaded with `local_files_only=true`; `bert-base-uncased` must
already exist in the Hugging Face cache. This is intentional so a formal run
cannot silently change model artifacts through a network download.

## Verify The Project

Run allocation tests:

```powershell
python -m unittest tests\test_allocations.py
```

Validate the main setup and existing results:

```powershell
python scripts\validate_setup.py `
  --config configs\expanded_sst2_bert_base_dirichlet.json

python scripts\validate_results.py `
  --config configs\expanded_sst2_bert_base_dirichlet.json `
  --results results\expanded_sst2_bert_base_dirichlet_v1
```

## Run The Main Experiment

The experiment is complete, so do not overwrite the v1 directory. Reproduce it
into a new versioned directory:

```powershell
python scripts\run_full_experiments.py `
  --config configs\expanded_sst2_bert_base_dirichlet.json `
  --out-dir results\expanded_sst2_bert_base_dirichlet_v2
```

The runner resumes at completed-method boundaries through each seed's
`checkpoint.json`. A changed protocol must use both a new config file and a new
output directory.

## Generate Tables And Figures

```powershell
python scripts\make_formal_comparison_table.py `
  --results results\expanded_sst2_bert_base_dirichlet_v1

python scripts\plot_expanded_results.py `
  --results results\expanded_sst2_bert_base_dirichlet_v1
```

Figures are written as 300-DPI PNG and vector PDF files under the run's
`figures/` directory.

## Method Names

| Code name | Meaning |
|---|---|
| `fedlora` | Full-layer, uniform-rank LoRA baseline |
| `resource_only` | Resource-capacity layer allocation |
| `fedra_random_partial` | Random partial-layer allocation |
| `fim_hla` | FIM/gradient-based HLA |
| `rgd_hla` | Randomized geometric HLA |
| `fedhello` | RGD warm start followed by periodic FIM allocation |
| `legend` | Paper-based LEGEND LCD reimplementation with dynamic depth and arithmetic rank distribution |
| `importance_resource_lora` | Global importance/resource allocation |
| `personalized_cost_aware_lora` | PCLA-v1 clustered layer allocation |
| `pcla_online_layer` | Online client-specific layer screening |
| `random_component_alternating` | Matched-budget random LoRA-component control |
| `pcla_component_alternating` | Online component allocation with alternating A/B training |
| `random_component_warmstart` | PCLA 1.2 matched warm-start random control |
| `pcla_component_warmstart` | PCLA 1.2 fixed warm start followed by component allocation |

## Reporting Rules

- Use `results/expanded_sst2_bert_base_dirichlet_v1` for the current main table.
- Report mean and sample standard deviation across all three seeds.
- Do not call this an exact numerical reproduction of the Fed-HeLLo paper. It is
  a controlled mechanism reproduction on a different dataset and scale.
- Do not claim that PCLA-v1 learned four meaningful clusters. Only two were used
  and they mainly align with resource levels.
- Do not report only the final round. Include convergence or late-round
  stability because several methods fluctuate under strong Non-IID data.
- Never treat files in `archive/` as current final evidence.

Detailed experiment interpretation is in [`docs/RESULTS.md`](docs/RESULTS.md).
Paper-derived assumptions are in
[`docs/FEDHELLO_REPRODUCTION_NOTES.md`](docs/FEDHELLO_REPRODUCTION_NOTES.md).
The three subsequent screening stages are documented in
[`docs/PCLA_ONLINE_LAYER_SCREENING_V1.md`](docs/PCLA_ONLINE_LAYER_SCREENING_V1.md),
[`docs/PCLA_ONLINE_EXPLORATION_ABLATION_V1.md`](docs/PCLA_ONLINE_EXPLORATION_ABLATION_V1.md),
and
[`docs/PCLA_COMPONENT_SCREENING_V1.md`](docs/PCLA_COMPONENT_SCREENING_V1.md).
The fixed warm-start follow-up and its rejected hypothesis are documented in
[`docs/PCLA_COMPONENT_WARMSTART_SCREENING_V1.md`](docs/PCLA_COMPONENT_WARMSTART_SCREENING_V1.md).
The corrected globally label-preserving client partition and the formal
component rerun are documented in
[`docs/DATA_PARTITION_CORRECTION_V1.md`](docs/DATA_PARTITION_CORRECTION_V1.md).
The 20-client, 10-selected-client, 30-round main component experiment is
documented in
[`docs/PCLA_LARGE_SCALE_CONSTRAINED_V1.md`](docs/PCLA_LARGE_SCALE_CONSTRAINED_V1.md).
