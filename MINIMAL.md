# LeJEPA: Minimal Example
*Author: Randall Balestriero, Yann LeCun*  
*Date: 2025-11-20*
---

This page explains a minimal implementation of [**LeJEPA**](https://arxiv.org/abs/2511.08544)  using a Vision Transformer (ViT) backbone and the [ImageNette](https://github.com/fastai/imagenette) dataset. The code builds upon the code snippets provided in the paper to provide a simplified version of the full method, designed for easy experimentation and extension, with the following core benefits:
- 130 lines total
- Wandb logging
- ViT and Imagenette (inet10)
- no teacher-student, stop-gradient, ...
- online linear probing
- SOTA top1 accuracy

Before reading more, make sure you have the minimal dependencies installed 
```bash
pip install torch torchvision timm wandb hydra-core datasets huggingface-hub
```
## Table of Contents
1. [Why LeJEPA?](#why-lejepa)
2. [The Code](#the-code)
3. [Case-Study: ViT/s-8 + Imagenette](#case-study-vits-8--imagenette)
4. [Reference](#reference)
---
## Why LeJEPA?

*TLDR: If you're pretraining self-supervised or foundation models, LeJEPA gives you SOTA results with a fraction of the complexity, backed by actual theory instead of trial-and-error heuristics.*


LeJEPA represents a fundamental rethinking of self-supervised learning. Instead of adding more and more heuristics to existing solutions, we erased the blackboard and re-designed a JEPA from scratch with theory-backed first principles. Here are some key benefits:
- **simple**: the core method takes ~50 lines of code, our complete example below is about 130 lines total. LeJEPA has a single hyperparameter (λ). No stop-gradients, no teacher-student networks, no register tokens, ...
- **provable**: LeJEPA's design is deeply anchored in a few theoretical results ensuring that it uniquely minimizes downstream task risk--we finally have a unified mathematical language to study JEPAs!
- **reliable**: LeJEPA works Out-of-the-Box across 60+ architectures (ResNets, ViTs, ConvNets), 10+ datasets, and scales from small models to 1.8B parameters. No painful hyperparameter tuning required.
- **informative loss**: the training Loss actually means something--boasting 94%+ Spearman correlation between training loss and downstream performance—you can finally do model selection without labeled validation data

Because of those benefits, it becomes trivial to perform in-domain pretraining to outperform massive frontier models--even in few-shot or with full finetuning. And because (i) LeJEPA's cross-validation grid is much smaller than other methods, (ii) LeJEPA works with very small mini-batch sizes or datasets, and (iii) LeJEPA's training loss is informative , you can finally pretrain without having access to thousands of high-end GPUs.

## The Code

The obvious yet necessary first part is to import all of our modules

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import timm, wandb, hydra, tqdm
from omegaconf import DictConfig
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
```
which include your typical PyTorch utilities, [wandb](https://wandb.ai/site/), [timm](https://github.com/huggingface/pytorch-image-models) and [HuggingFace](https://huggingface.co/) for conveniencce around logging, dataset and model definition (oh yea, LeJEPA doesn't need you to implement a custom model architecture or parameter initialization, *any will work*).

Next, let's define our backbone with projector, our dataset, and our infamous SIGReg objective (the core component of LeJEPA). One may notice a small difference in the implementation of SIGReg from the paper's one: we leverage the symmetric property of the ECF/CF improve the quadrature efficient (integrate on `[0, t_max]` and doubling, instead of integrating on `[-t_max, t_max]`), *improved quadrature for free*:

```python
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split, V=1):
        self.V = V
        self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test = v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), item["label"]

    def __len__(self):
        return len(self.ds)

```
And that's all we need to define before creating our main function that will assemble all those components and iterate through training and validation steps! Note that we put some generic hyper-parameters in the above that will not impact performance, e.g., `drop_path_rate`. Here is the main loop:

```python

@hydra.main(version_base=None)
def main(cfg: DictConfig):
    wandb.init(project="LeJEPA", config=dict(cfg))
    torch.manual_seed(0)

    train_ds = HFDataset("train", V=cfg.V)
    test_ds = HFDataset("validation", V=1)
    train = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=8
    )
    test = DataLoader(test_ds, batch_size=256, num_workers=8)

    # modules and loss
    net = ViTEncoder(proj_dim=cfg.proj_dim).to("cuda")
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 100)).to("cuda")
    sigreg = SIGReg().to("cuda")
    # Optimizer and scheduler
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled="cuda" == "cuda")
    # Training
    for epoch in range(cfg.epochs):
        net.train(), probe.train()
        for vs, y in tqdm.tqdm(train, total=len(train)):
            with autocast("cuda", dtype=torch.bfloat16):
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep)
                loss = lejepa_loss + probe_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            wandb.log(
                {
                    "train/probe": probe_loss.item(),
                    "train/lejepa": lejepa_loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/inv": inv_loss.item(),
                }
            )

        # Evaluation
        net.eval(), probe.eval()
        correct = 0
        with torch.inference_mode():
            for vs, y in test:
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    correct += (probe(net(vs)[0]).argmax(1) == y).sum().item()
        wandb.log({"test/acc": correct / len(test_ds), "test/epoch": epoch})
    wandb.finish()


if __name__ == "__main__":
    main()
```
And that all you need to pretraining your (Le)JEPA! Let's look below at an actual example.

## Case-Study: ViT/s-8 + Imagenette

The code provided above builds upon timm and HuggingFace, so it is a matter of seconds to launch training on any dataset and backbone! We are considering here ViT/s-8 and Imagenette (inet10) since it is hard enough to be interesting but fast enough to play with. *Expect about 2h total runtime to reproduce the below experiments on a single GPU*.

First, let's launch training! Logs will be in your Wandb project called `LeJEPA`
We recommend launching the script with the following hyper-parameters
```python
python mnist.py +lamb=0.02 +V=4 +proj_dim=16 +lr=2e-3 +bs=256 +epochs=800
```

As you will quickly see, training curves are extremely smooth and stable even in bf16 precision **without having to resort to gradient clipping or any particular heuristic**, and remember, **this is all without stop-gradient, teacher-students, and the likes!** Here is how it looks:
| | |
|:-------------------------:|:-------------------------:|
| **Prediction/Invariance loss** | **SIGReg loss** |
|<img width="500" alt="invariance loss" src="https://i.imgur.com/aCVn2WH.png">   | <img width="500" alt="sigreg loss" src="https://i.imgur.com/fFHp4t2.png">|
| **LeJEPA (pred $\times (1-lamb)$ + SIGReg $\times lamb$ ) loss** | **Online probe loss** 
|<img width="500" alt="LeJEPA loss" src="https://i.imgur.com/lc8vn37.png">  |<img width="500" alt="online probe loss" src="https://i.imgur.com/ejLcW4o.png">|

And last but not least, the online accuracy (per-epoch) is also very stable--depicted here with relative walltime in the x-axis:
| |
|:-------------------------:|
| **Online test accuracy** |
|<img width="1604" alt="test acc" src="https://i.imgur.com/iayd0IB.png">   |

The choice of batch-size and epochs wasn't arbitrary, it coincides with a benchmark provided by Lightly that we can use for a ballpark reference (they use k-NN probing while we use linear probing):


| | |
|:-------------------------:|:-------------------------:|
| [**Lightly benchmark**](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html) | **LeJEPA** |
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://i.imgur.com/hu5qIG2.png">   | 90.7\%, 200 Min|





## Reference
We hope that our findings and LeJEPA will be useful to you! Here is our BibTeX if you would like to cite our work:
```
@misc{balestriero2025lejepaprovablescalableselfsupervised,
      title={LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics}, 
      author={Randall Balestriero and Yann LeCun},
      year={2025},
      eprint={2511.08544},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.08544}, 
}
```

## Questions or suggestions? 
Open an issue or reach out!
