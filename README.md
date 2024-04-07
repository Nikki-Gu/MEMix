# MEMix

This is the official code of the [paper](https://ieeexplore.ieee.org/document/10446492): **Facial Micro-motion-aware Mixup for Micro-expression Recognition**, ICASSP 2024.

## Experiment Results

To further validate the performance and generalizability of MixMeFormer, we conducted additional experiments on the SAMM (3-class) and MMEW (4-class) datasets.

On SAMM, we compared with a wide range of 8 state-of-the-art methods as below. Our MixMeFormer achieves the highest accuracy and F1-score.

| Method                                                       | Year | Type | Acc(%)       | F1-score        |
| ------------------------------------------------------------ | :--: | :--:| :--: | :--: |
| STSTNet                                                     | 2019| CNN | 68.10 | 0.6588 |
| MiMANet                                                     | 2021| CNN | 76.60 | 0.7640|
| MERSiamC3D | 2021 | CNN | 72.80 | 0.7475 |
| MAPNet | 2022 | CNN | 86.50 | 0.8160 |
| AMAN | 2022 | CNN | 68.85 | 0.6682 |
| AU-GACN | 2020 | GCN | 70.20 | 0.4330 |
| MMNet | 2022 | CNN+Transformer | 90.22 | 0.8391 |
| FRL-DGT | 2023 | CNN+Transformer | - | 0.7720 |
| **MixFormer(ours)** | **2023** | **CNN+Transformer** | **90.23** | **0.8477** |

On the MMEW dataset, MixMeFormer also shows superior performance over MMNet.

| Method    | Acc(%)       | F1-score        |
| ------------------------------------------------------------ | :--: | :--: |
| MMNet | 87.45 | 0.8635 |
| **MixFormer(ours)** | **88.59** | **0.8698** |

Besides, we compared our MEMix with six mixup augmentation methods, including three newly proposed mixup methods [Manifold Mixup](https://github.com/DaikiTanak/manifold_mixup), [Remix](https://github.com/agaldran/balanced_mixup) and [MixAugment](https://github.com/dongdong69/MixAugmentation). This experiment was conducted based on the ViT-B architecture on the 5 classes CASME II dataset. Our MEMix achieves the highest improvements in both accuracy and F1-score.

| Method                                                       | Acc(%)       | F1-score        |
| ------------------------------------------------------------ | :--: | :--: |
| baseline                                                     | 73.98        | 0.7200          |
| [Mixup](https://github.com/facebookresearch/mixup-cifar10)   | 82.11(+8.13) | 0.8145(+0.0945) |
| [CutMix](https://github.com/clovaai/CutMix-PyTorch)          | 82.93(+8.95) | 0.8133(+0.0933) |
| [Manifold Mixup](https://github.com/DaikiTanak/manifold_mixup) | 83.33(+9.35) | 0.8179(+0.0979) |
| [TransMix](https://github.com/Beckschen/TransMix)            | 83.74(+9.76) | 0.8048(+0.0848) |
| [Remix](https://github.com/agaldran/balanced_mixup)          | 82.52(+8.54) | 0.8186(+0.0986) |
| [MixAugment](https://github.com/dongdong69/MixAugmentation)  | 83.33(+9.35) | 0.8254(+0.1054) |
| **MEMix(ours)** | **85.37(+11.39)** | **0.8365(+0.1165)** |


### Hyperparameter Experiments

Additionally, we conducted experiments studying the impacts of two key hyperparameters $K$ and $\alpha_k$ in our proposed MEMix.

$K$ determines the number of patches selected to construct the mixing mask $M$. As shown below, performance peaks at $K=40$ and then decreases as $K$ becomes too large. This aligns with our motivation to only mix the most salient motion regions.

**The experimental results of varying $K$ are summarized in the table below:**
|  K   |   1    |   40   |   79   |  118   |  157   |  196   |
| :--: | :----: | :----: | :----: | :----: | :----: | :----: |
| Acc(%)  | 0.7805 | **0.8699** | 0.8618 | 0.8577 | 0.8496 | 0.8333 |

We also studied the impact of the hyperparameter $\alpha_k$, which controls the beta distribution for sampling $K$. As can be seen, $\alpha_k=2.0$ achieves the optimal accuracy, while too small or too large values degrade the performance.

**The experimental results of varying $\alpha_k$ are summarized below:**

|   $\alpha_k$   |  1.0  |  1.5  |  2.0  |  2.5  |  3.0  |
| :---------: | :---: | :---: | :---: | :---: | :---: |
| Acc(%) | 84.96 | 85.37 | **89.84** | 86.59 | 86.59 |

## Citation
If you find this repository useful, please cite the paper:
```
@INPROCEEDINGS{10446492,
  author={Gu, Zhuoyao and Pang, Miao and Xing, Zhen and Tan, Weimin and Jiang, Xuhao and Yan, Bo},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Facial Micro-Motion-Aware Mixup for Micro-Expression Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={8060-8064},
  keywords={Face recognition;Computational modeling;Semantics;Speech recognition;Signal processing;Transformers;Data models;Micro-expression recognition;Data augmentation;Vision transformer},
  doi={10.1109/ICASSP48485.2024.10446492}}
```
