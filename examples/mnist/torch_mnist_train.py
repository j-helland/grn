import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchmetrics

import grn


@grn.job()
def train_job(epochs: int, batch_size: int) -> 'torch.nn.Module': 
    def _get_loader(train=True):
        return torch.utils.data.DataLoader(
            tv.datasets.MNIST(
                root='./',
                download=True,
                train=train, 
                transform=tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.0,), (1.0,))])),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2)

    train_loader, test_loader = _get_loader(True), _get_loader(False)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    accuracy = torchmetrics.Accuracy().cuda()

    for epoch in range(epochs):
        accuracy.reset()
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy(logits.softmax(dim=1), y)
        
        train_acc = accuracy.compute()

        accuracy.reset()
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                accuracy(model(x).softmax(dim=1), y)
        
        test_acc = accuracy.compute()

        print(f'Epoch {epoch + 1} / {epochs} | train acc {train_acc:.4f} | test acc {test_acc:.4f}')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    trained_model = train_job(
        epochs=args.epochs, 
        batch_size=args.batch_size)
    