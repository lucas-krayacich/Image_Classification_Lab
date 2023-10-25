import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from vanilla_model import vanilla_model

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Basic options
    # parser.add_argument('-l', type=str, default='./encoder.pth')
    # parser.add_argument('-gamma', type=float)
    parser.add_argument('-b', "--batch_size", type=int, default=20)
    # parser.add_argument('-e', "--epochs", type=int)
    # parser.add_argument('-s', type=str)
    parser.add_argument('-p', type=str)
    parser.add_argument('-cuda', type=str)
    # training options
    parser.add_argument('-model_pth', default='',
                        help='model weights')
    parser.add_argument('-lr', type=float, default=0.001)

    args = parser.parse_args()
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.b,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.b,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    vanilla_model = vanilla_model()
    vanilla_model.load_state_dict(torch.load(args.model_pth))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = vanilla_model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
