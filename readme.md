# ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration (CVPR 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2603.00906-b31b1b.svg)](https://arxiv.org/abs/2603.00906) [![Conference](https://img.shields.io/badge/CVPR-2026-blue)](https://cvpr.thecvf.com/)

---

## Step 1: Train the CNN Backbone

```bash
cd Net_train/[task_type]

# Stage 1: Train the network with offset prediction module
bash train_stage1.sh

# Evaluate the Stage 1 model and collect predicted offsets from LSS
python eval_offset.py

# Stage 2: Train the network with fixed offset
bash train_stage2.sh

# (Optional) Evaluate the Stage 2 model
python eval.py
```

---

## Step 2: Convert CNN to LUT

```bash
python net2LUT.py
```

This script converts a trained CNN model into a set of LUTs for efficient inference.

---

## Step 3: Evaluate LUT Performance

```bash
cd LUT_test/[task_type]
python eval_LUT.py
```

This runs the final evaluation using the converted LUTs.

---

> 🔁 Replace `[task_type]` with one of: `sr`, `denoising`, or `deblocking`.

---

## Project Structure

```text
ShiftLUT/
├── Net_train/
│   ├── sr/              # Super-resolution training code
│   ├── denoising/       # Denoising training code
│   └── deblocking/      # Deblocking training code
│
├── models/              # Trained CNN models
├── net2LUT.py           # Convert trained network to LUT
│
├── LUT_test/
│   ├── sr/              # Super-resolution LUT evaluation
│   ├── denoising/       # Denoising LUT evaluation
│   └── deblocking/      # Deblocking LUT evaluation
```

# Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{zeng2026shiftlut,
  title={ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration},
  author={Zeng, Xiaolong and Yu, Yitong and Xiong, Shiyao and Hao, Jinhua and Sun, Ming and Zhou, Chao and Wang, Bin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026},
  url={https://arxiv.org/abs/2603.00906}
}
```

## Acknowledgement
This work is based on the following works, thank the authors a lot.

[SR-LUT](https://github.com/yhjo09/SR-LUT)

[MuLUT](https://github.com/ddlee-cn/MuLUT/tree/main)

[TinyLUT](https://github.com/Jonas-KD/TinyLUT)
