import torch.optim as optim
import torch.nn as nn
from vanilla_model import vanilla_model
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


def train(n_epochs, optimizer, model, loss_fn, save_path, trainloader):

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

            torch.save(vanilla_model.state_dict(), save_path)

    print('Finished Training')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Basic options
    # parser.add_argument('-l', type=str, default='./encoder.pth')
    # parser.add_argument('-gamma', type=float)
    # parser.add_argument('-b', "--batch_size", type=int, default=20)
    parser.add_argument('-e', "--epochs", type=int)
    parser.add_argument('-s', type=str)
    parser.add_argument('-p', type=str)
    parser.add_argument('-cuda', type=str)
    # training options
    parser.add_argument('-save_dir', default='',
                        help='Directory to save the model')
    parser.add_argument('-lr', type=float, default=0.001)
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vanilla_model.parameters(), lr=args.lr, momentum=0.9)
    save_dir = args.s

    train(
        n_epochs=args.epochs,
        optimizer=optimizer,
        model=vanilla_model,
        loss_fn=criterion,
        save_path=save_dir,
        trainloader=trainloader
    )
