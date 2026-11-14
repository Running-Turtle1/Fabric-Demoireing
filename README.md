## Environment

Create a Python environment with PyTorch installed for your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## PRISM Dataset

Download and extract PRISM from [Hugging Face](https://huggingface.co/datasets/running-Turtle/PRISM):

```bash
pip install -U huggingface_hub
huggingface-cli download running-Turtle/PRISM \
    --repo-type dataset \
    --local-dir dataset/PRISM_download

cat dataset/PRISM_download/train_chunks/PRISM_train.zip.part-* \
    > dataset/PRISM_download/PRISM_train.zip

unzip dataset/PRISM_download/PRISM_train.zip -d dataset
unzip dataset/PRISM_download/PRISM_test.zip -d dataset
```

## Pretrained Weights

Download the pretrained FaDeNet checkpoint from [Google Drive](https://drive.google.com/file/d/18n2ce1Q-XvHH7F9bztqdxIQDi1CTt_ri/view?usp=sharing).

```bash
mkdir -p weights/FaDeNet
cp /path/to/best_psnr.checkpoint weights/FaDeNet/
```

## PRISM Evaluation

```bash
python -m scripts.FaDeNet.test \
  --config configs/FaDeNet/prism_config.yaml \
  --load_path weights/FaDeNet/best_psnr.checkpoint \
  --workers 4 \
  --save_dir outputs/reproduce/FaDeNet/prism
```

> **Tip:** The released FaDeNet checkpoint was obtained from a separate retraining run, so its PSNR may differ slightly from the value reported in Table 1 (by approximately **0.01 dB**).

## Training

```bash
python -m scripts.FaDeNet.train \
  --config configs/FaDeNet/prism_config.yaml
```

## Citation

If you use PRISM, please cite:

```bibtex
@inproceedings{wei2026fabric,
  title     = {Fabric Image Demoir{\'e}ing Benchmark from Synthesis to Restoration},
  author    = {Wei, Pengchao and Guo, Xiaojie},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```
