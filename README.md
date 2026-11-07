# AlexNet from Scratch vs PyTorch

A custom implementation of the original AlexNet architecture (96/256/384/384/256 filters), compared against PyTorch's built-in `torchvision.models.alexnet` on CIFAR-10.

## Overview

- **Custom AlexNet:** built layer by layer with `nn.Module` (original paper filter sizes)
- **PyTorch AlexNet:** `torchvision.models.alexnet(weights=None)` with the last layer changed to 10 classes
- **Dataset:** CIFAR-10, images resized to 227x227
- **Both models trained from scratch** (no pretrained weights) with identical settings

## Training Setup

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 |
| Input size | 227 x 227 |
| Augmentation | Random horizontal flip |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss | CrossEntropyLoss |
| Epochs | 10 |
| Hardware | Google Colab T4 GPU |

## Architecture (My AlexNet)

| Layer | Details |
|-------|---------|
| Conv1 | 96 filters, 11x11, stride 4 + ReLU + MaxPool(3, stride 2) |
| Conv2 | 256 filters, 5x5, padding 2 + ReLU + MaxPool(3, stride 2) |
| Conv3 | 384 filters, 3x3, padding 1 + ReLU |
| Conv4 | 384 filters, 3x3, padding 1 + ReLU |
| Conv5 | 256 filters, 3x3, padding 1 + ReLU + MaxPool(3, stride 2) |
| FC1 | 9216 -> 4096 + ReLU |
| FC2 | 4096 -> 4096 + ReLU |
| FC3 | 4096 -> 10 |

PyTorch's version uses 64/192/384/256/256 filters, 11x11 conv with padding 2, adaptive average pooling and dropout, which explains the small parameter difference.

## Results

| Metric | My AlexNet | PyTorch AlexNet |
|--------|-----------|-----------------|
| Total parameters | 58,322,314 | 57,044,810 |
| Test accuracy | 82.24% | 82.83% |

Both models land within about 0.6% of each other, showing the custom implementation behaves like the reference.

## Project Structure

```
alexnet-vs-pytorch/
├── alexnet_comparison.ipynb   # Training + comparison notebook
└── README.md
```

## Setup

```bash
git clone https://github.com/krishtyagi0109-pixel/alexnet-implementation-benchmark.git
cd alexnet-implementation-benchmark
pip install torch torchvision
```

Open the notebook in Google Colab (GPU runtime recommended) and run all cells.

## Key Findings

- The custom AlexNet reaches **82.24%** vs **82.83%** for PyTorch's AlexNet on CIFAR-10.
- The parameter gap (about 1.3M) comes from different filter counts in the early conv layers.
- The small accuracy difference is within normal run-to-run variance, since both were trained from scratch with random initialization.

## Possible Improvements

- Add Dropout to the custom FC layers (the original paper and torchvision both use it)
- Add Local Response Normalization as in the original paper
- Run multiple seeds to average out variance
- Plot loss curves for both models

## Reference

- [ImageNet Classification with Deep CNNs (Krizhevsky et al., 2012)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

## License

MIT
