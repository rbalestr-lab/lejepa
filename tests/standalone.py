"""
Standalone visualization script for diffusiondb dataset.

This script imports library components to visualize images from the diffusiondb dataset.
"""
import torch
from lejepa.univariate.likelihood import NLL
from lejepa.multivariate.slicing import SlicingUnivariateTest


if __name__ == "__main__":
    from datasets import load_dataset  # Datasets not included!!!
    import matplotlib.pyplot as plt

    # SECURITY NOTE: trust_remote_code=True removed - use only trusted datasets
    dataset = iter(
        load_dataset(
            "poloclub/diffusiondb",
            split="train",
            streaming=True,
            # trust_remote_code=True,  # REMOVED: Security risk
        ).sort("image_nsfw")
    )
    fig, axs = plt.subplots(6, 6, figsize=(15, 15))
    fig2, axs2 = plt.subplots(6, 6, figsize=(15, 15))

    for ax, ax2 in zip(axs.flatten(), axs2.flatten()):
        print("DONE")
        img = next(dataset)["image"]
        ax.imshow(img.resize((512, 512)), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.imshow(img.resize((64, 64)), interpolation="nearest")
        ax2.set_xticks([])
        ax2.set_yticks([])
    fig.tight_layout()
    fig.savefig("high_fake.png")
    fig2.tight_layout()
    fig2.savefig("low_fake.png")

    # SECURITY: removed trust_remote_code=True
    dataset = iter(
        load_dataset(
            "ILSVRC/imagenet-1k",
            split="train",
            streaming=True,
            # trust_remote_code=True,  # REMOVED: Security risk
        )
    )
    fig, axs = plt.subplots(6, 6, figsize=(15, 15))
    fig2, axs2 = plt.subplots(6, 6, figsize=(15, 15))

    for ax, ax2 in zip(axs.flatten(), axs2.flatten()):
        print("DONE")
        img = next(dataset)["image"]
        ax.imshow(img.resize((512, 512)), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.imshow(img.resize((64, 64)), interpolation="nearest")
        ax.set_xticks([])
        ax2.set_yticks([])

    fig.tight_layout()
    fig.savefig("high_real.png")
    fig2.tight_layout()
    fig2.savefig("low_real.png")

    # SECURITY: removed trust_remote_code=True
    dataset = load_dataset(
        "poloclub/diffusiondb",
        split="train",
        streaming=True,
        # trust_remote_code=True,  # REMOVED: Security risk
    )
    img = next(iter(dataset))["image"]
    plt.imshow(img.resize((512, 512)))
    plt.savefig("real.png")
    plt.close()

    uni_test = NLL()
    multi_test = SlicingUnivariateTest(
        dim=(1, 2), univariate_test=uni_test, num_slices=100
    )
    print(multi_test(torch.randn(10, 32, 128)))
