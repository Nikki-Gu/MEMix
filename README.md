# MEMix

This is the official code of the paper: 

**Facial Micro-motion-aware Mixup for Micro-expression Recognition**

## Experiment Results
In addition to the three fundamental mixup methods, we also conducted comparisons with three newly proposed mixup augmentation methods: [TransMix](https://github.com/Beckschen/TransMix), [Remix](https://github.com/agaldran/balanced_mixup) and [MixAugment](https://github.com/dongdong69/MixAugmentation). 

This experiment was conducted based on the ViT-B architecture on the 5 classes CASME II dataset. Our MEMix achieves the highest improvements in both accuracy and F1-score.
| Method                                                       | Acc(%)       | F1-score        |
| ------------------------------------------------------------ | ------------ | --------------- |
| baseline                                                     | 73.98        | 0.7200          |
| [Mixup](https://github.com/facebookresearch/mixup-cifar10)   | 82.11(+8.13) | 0.8145(+0.0945) |
| [CutMix](https://github.com/clovaai/CutMix-PyTorch)          | 82.93(+8.95) | 0.8133(+0.0933) |
| [Manifold Mixup](https://github.com/DaikiTanak/manifold_mixup) | 83.33(+9.35) | 0.8179(+0.0979) |
| [TransMix](https://github.com/Beckschen/TransMix)            | 83.74(+9.76) | 0.8048(+0.0848) |
| [Remix](https://github.com/agaldran/balanced_mixup)          | 82.52(+8.54) | 0.8186(+0.0986) |
| [MixAugment](https://github.com/dongdong69/MixAugmentation)  | 83.33(+9.35) | 0.8254(+0.1054) |
| MEMix(ours) | **85.37(+11.39)** | **0.8365(+0.1165)** |




| Method                                                       | Year | Type | Acc(%)       | F1-score        |
| ------------------------------------------------------------ | ------------ | --------------- | ------------ | --------------- |
| STSTNet                                                     | 2019| CNN | 68.10 | 0.6588 |
| MiMANet                                                     | 2021| CNN | 76.60 | 0.7640|
| MERSiamC3D | 2021 | CNN | 72.80 | 0.7475 |
| MAPNet | 2022 | CNN | 86.50 | 0.8160 |
| AMAN | 2022 | CNN | 68.85 | 0.6682 |
| AU-GACN | 2020 | GCN | 70.20 | 0.4330 |
| MMNet | 2022 | CNN+Transformer | 90.22 | 0.8391 |
| FRL-DGT | 2023 | CNN+Transformer | - | 0.7720 |
| MixFormer(ours) | 2023 | CNN+Transformer | **90.23** | **0.8477** |



