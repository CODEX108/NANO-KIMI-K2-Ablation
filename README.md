## NANO-KIMI-K2-Ablation
Optimizer and Data Ablation StudyThis repository serves as a dedicated **ablation study** branch of the NANO-KIMI-K2 project.
Its sole purpose is to quantitatively analyze the impact of critical architectural choices the **optimizer** and **pre-training data quality** on LLM training stability and convergence.

The goal is to provide empirical evidence justifying the choice of advanced techniques (MuonClip, Curated Data) over common baselines (AdamW, General Web Data).

### Ablation FocusThe project specifically compares two key components that govern the efficiency and stability of large-scale Transformer pre-training.

| Ablation Variable | Baseline Component | Advanced Component | Measured Impact on |
| --- | --- | --- | --- |
| **Optimizer** | **AdamW** | **MuonClip** | Loss Stability, Token Efficiency, Memory Footprint. |
| **Data Quality** | **FineWeb** (Broad) | **FineWeb-Edu + Curated Mix** (Targeted) | Initial Loss Ceiling, Validation Perplexity, Knowledge Retention. |

## Key FindingsThe ablation experiments confirmed the benefits of the advanced configuration:

* **MuonClip vs. AdamW:** MuonClip (with QK-Clip) yields a **significantly smoother loss trajectory** and **zero catastrophic loss spikes** throughout training steps. This stability is achieved by actively controlling attention logit magnitude at the weight level, which is not possible with standard AdamW.
* **Curated Data:** Training on a refined mix (FineWeb-Edu, Cosmopedia) demonstrates an **improved initial convergence rate** and is expected to lead to a superior final perplexity ceiling, validating the higher informational density of the curated set.

## Repository StructureThis repository uses core components from the NANO-KIMI-K2 architecture, with specialized files enabling the comparative study.

| File | Role in Ablation |
| --- | --- |
| `optimizer.py` | Contains the **MuonClip** implementation for comparison against standard AdamW. |
| `finewebdataset.py` | Dedicated loader for the **FineWeb** baseline data. |
| `dataset.py` | Handles the loader for the **Curated Mix** (FineWeb-Edu, Cosmopedia, etc.). |
| `train.py` | Main script for running controlled experiments across different optimizer/data configs. |
| `inference.py` | Used to test the final state of ablated models. |
| `config.py` / `models.py` | Defines the fixed MoE and MLA architecture used for all ablation runs. |

##🚀 Quick ExecutionTo reproduce the core comparison (AdamW/FineWeb baseline vs. MuonClip/Curated):

1. **Clone & Setup:**
```bash
git clone https://github.com/CODEX108/NANO-KIMI-K2-Ablation.git
cd NANO-KIMI-K2-Ablation
pip install -r requirements.txt

```


2. **Run Ablation Experiments:** Use specific configuration flags (or config files if provided) in `train.py` to switch between the two study arms.
```bash
# Example: Run Baseline (AdamW on FineWeb)
python train.py \
    --dataset fineweb \
    --num_samples 10000 \
    --batch_size 2 \
    --max_steps 2000 \
    --save_interval 500

```

3. **Inference**
```# Basic Interactive Mode
!python inference.py checkpoints/latest.pt --interactive

========================================================

!python inference.py \
    --prompt "Once upon a time" \
    --max_tokens 300 \
    --temperature 0.8 \
    --device cuda \
    ./latest.pt
```
