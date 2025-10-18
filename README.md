```markdown
# Multimodal Fake News Detection — Complete Starter Project

This repository contains a complete starter scaffold for a university project on Multimodal Fake News Detection (text + image + propagation features). It includes:

- Data preprocessing for FakeNewsNet / LIAR / Fakeddit style data
- PyTorch Dataset and Dataloaders
- Text baseline (BERT), Image baseline (ResNet), Propagation feature baseline (XGBoost)
- Fusion models (late fusion ensemble + early fusion)
- Training/evaluation scripts, robustness tests, explanation hooks
- Colab-ready notebook to run a text-only baseline quickly
- Config files for reproducible experiments

Ethics & data:
- You must respect dataset licenses and Twitter API Terms of Service when re-downloading tweets or images.
- Anonymize user IDs before public release.
- Document ethical considerations in your report.

Quick start (local, GPU recommended):
1. Create Python venv and install requirements:
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt

2. Prepare dataset:
   - If using FakeNewsNet: download and extract repository (see links in this README).
   - Run preprocessing to create dataset CSV (examples shown in src/data/preprocess.py).

3. Run text baseline training (example):
   python src/train_text.py --config configs/train_text.yaml

4. Run image baseline (if images available):
   python src/train_image.py --config configs/train_image.yaml

5. Run fusion training:
   python src/train_fusion.py --config configs/train_fusion.yaml

Files of interest:
- src/data/preprocess.py — generate a single annotated CSV from raw dataset.
- src/data/dataset.py — PyTorch Dataset for multimodal samples.
- src/models.py — text/image/fusion model classes.
- src/train_text.py, src/train_image.py, src/train_fusion.py — training scripts.
- src/eval.py — evaluation & robustness tests.
- notebooks/text_baseline_colab.md — Colab-ready text baseline notebook.

If you want:
- I can convert the notebook into an executed Colab notebook (.ipynb) and link a runnable Colab.
- I can expand any model (cross-attention module, GNN propagation model) into full code.
- I can add WandB logging or Dockerfile for deployment.

Team / timeline suggestions are in the project writeup (see top-level project plan in the original materials).
```
