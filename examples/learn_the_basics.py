import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


def datasets_and_dataloaders():

    training_data = datasets.FashionMNIST(
        "data",
        download=True,
        train=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    test_data = datasets.FashionMNIST(
        "data",
        download=True,
        train=False,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )


def main():
    device = "cuda" if torch.accelerator.is_available() else "cpu"
    device = "cpu"
    N = 1024 * 16

    t1 = time.time()
    m1 = torch.rand((N, N)).to(device)
    m2 = torch.rand((N, N)).to(device)

    print((m1 @ m2).sum())
    t2 = time.time()
    print(f"Tiempo de ejecución: {t2 - t1:.4f} segundos")


if __name__ == "__main__":
    main()
