import logging

import torchvision
import liblog
from logging import Logger

from dataclasses import dataclass
from os import getcwd
import torch
import torch.nn.functional as f
from torch import nn, optim
from torch.nn import Conv2d, MaxPool2d, Linear, Softmax
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter, writer


class NeuralNetwork(nn.Module):
    """Main neural network class."""

    conv1: Conv2d
    conv2: Conv2d
    conv3: Conv2d
    pool: MaxPool2d
    full: Linear

    def __init__(self) -> None:

        super(NeuralNetwork, self).__init__()

        """The constructor for torch.nn.modules.conv.Conv2d is as follows:

        def __init__(self,
             in_channels: int,
             out_channels: int,
             kernel_size: Tuple[int, ...],
             stride: Tuple[int, ...] = 1,
             padding: Tuple[int, ...] = 0,
             dilation: Tuple[int, ...] = 1,
             groups: int = 1,
             bias: bool = True,
             padding_mode: str = 'zeros') -> None

        There is a type mismatch error in the following two lines.
        (Tuple[int] expected, got int instead)
        I have tried to fix it; please alter if behaviour is unexpected.

        """

        self.conv1 = nn.Conv2d(3, 48, 3, 1)
        self.conv2 = nn.Conv2d(48, 48, 3, 1)
        self.batch1 = nn.BatchNorm2d(48)
        self.batch2 = nn.BatchNorm2d(48)
        self.drop1 = nn.Dropout(0.15)

        self.conv3 = nn.Conv2d(48, 96, 3, 1)
        self.conv4 = nn.Conv2d(96, 96, 3, 1)
        self.batch3 = nn.BatchNorm2d(96)
        self.batch4 = nn.BatchNorm2d(96)
        self.drop2 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(96, 192, 3, 1)
        self.conv6 = nn.Conv2d(192, 192, 3, 1)
        self.batch5 = nn.BatchNorm2d(192)
        self.batch6 = nn.BatchNorm2d(192)
        self.drop3 = nn.Dropout(0.3)


        self.pool = nn.MaxPool2d(2, 2)

        self.full = nn.Linear(192, 10)

        log_event(0, "init of neural network completed")

    def forward(self, x) -> Softmax:
        """Layers of the dataset with activation functions

        Args:
            x: the input data
        Returns:
            output: containing the predicted values of the inputted data
        """

        x = self.conv1(x)
        x = f.relu(x)
        x = self.batch1(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = self.batch2(x)
        x = self.pool(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = f.relu(x)
        x = self.batch3(x)
        x = self.conv4(x)
        x = f.relu(x)
        x = self.batch4(x)
        x = self.pool(x)
        x = self.drop2(x)

        x = self.conv5(x)
        x = f.relu(x)
        x = self.batch5(x)
        x = self.conv6(x)
        x = f.relu(x)
        x = self.batch6(x)
        x = self.drop3(x)

        x = x.view(x.size(0), -1)
        x = self.full(x)

        return x


@dataclass
class DataSet(object):
    """Object to contain loaded training and test data"""

    train: DataLoader
    test: DataLoader


class DeviceDataLoader():
    """Wrap deviceLoader to move to device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def log_event(level: int, message: str) -> None:
    """Writes to both log file and terminal (for neater code).

    Please use this function for logging unless you have a specific reason not
    to (e.g. if you need traceback information).

    Args:
        level: the type of log to write:
            -3: critical
            -2: error
            -1: warning
             0: info
             1: debug

        message: the string to write

    """

    if level == -3:
        log.critical(message)
        print("[critical]", message)
    if level == -2:
        log.error(message)
        print("[error]", message)
    if level == -1:
        log.warning(message)
        print("[warning]", message)
    if level == 0:
        log.info(message)
        print("[info]", message)
    if level == 1:
        log.debug(message)
        print("[debug]", message)


def get_shuffled_sets() -> DataSet:
    """Fetches and loads training and testing CIFAR10 datasets from the CWD.

    Training data and test data is downloaded from the Internet if it cannot
    be found locally. Both datasets are then loaded, with the training set
    split into 5 batches.

    Returns:
        DataSet: Object containing loaded training and test data

    """

    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2470, 0.2435, 0.2616))
        ])

    train_data: CIFAR10 = datasets.CIFAR10(  # loads CIFAR10 training Data
        root=getcwd(),
        train=True,
        download=True,
        transform=transformations
    )

    test_data: CIFAR10 = datasets.CIFAR10(  # loads CIFAR10 test Data\
        root=getcwd(),
        train=False,
        download=True,
        transform=transformations
    )

    # loads data into respective batch sizes and then shuffles the dataset
    data: DataSet = DataSet(
        DataLoader(train_data, batch_size=50, shuffle=True),
        DataLoader(test_data, batch_size=50, shuffle=True)
    )
    print()
    return data


def to_device(data, device):
    """Transfers a tensor to a device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# TODO: finish docstring when func is completed
def train_nn(trainLoader: DataLoader, device: str, test_data: DataLoader, writer: SummaryWriter) -> NeuralNetwork:

    trainLoader = DeviceDataLoader(trainLoader, device)
    my_nn = to_device(NeuralNetwork(), device)

    lossFn = nn.CrossEntropyLoss()

    for i, data in enumerate(trainLoader):
        images, _ = data
        writer.add_graph(my_nn, images)
        break

    runLoss = 0.0
    runCorrect = 0

    for epoch in range(125):

        if epoch < 80:
            optimiser = optim.SGD(my_nn.parameters(), lr=1e-3, momentum=0.9)
        elif epoch < 120:
            optimiser = optim.SGD(my_nn.parameters(), lr=1e-3/2, momentum=0.9)
        elif epoch < 200:
            optimiser = optim.SGD(my_nn.parameters(), lr=1e-3/3, momentum=0.9)
        else:
            optimiser = optim.SGD(my_nn.parameters(), lr=1e-3/4, momentum=0.9)

        for i, dat in enumerate(trainLoader):
            inputs, labels = dat

            # forward pass
            outputs = f.softmax(my_nn(inputs))

            # backward pass
            loss = lossFn(outputs, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # print statistics
            runLoss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            runCorrect += (predicted == labels).sum().item()
            if (i+1) % 500 == 0:    # every 500 iters to change this change all similar values inside the if statement below
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, runLoss/500))

                writer.add_scalar('training loss', runLoss /
                                  500, (epoch + 1) + i)
                writer.add_scalar('accuracy', runCorrect /
                                  500, (epoch + 1) + i)
                runCorrect = 0
                runLoss = 0.0

        if (epoch+1) % 10 == 0:
            message = "Epoch #" + \
                str(epoch+1) + " Accuracy:" + \
                str(test_nn(test_data, my_nn)) + "%"
            log_event(0, message)

    writer.flush()

    return my_nn


# TODO: finish docstring when func is completed
def test_nn(test_data: DataLoader, my_nn: NeuralNetwork) -> float:

    log_event(0, "starting test of NN")

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:

            # get the images and send data to cuda or cpu device
            images, labels = data
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            images, labels = images.to(device), labels.to(device)

            outputs = my_nn(images)
            # print(outputs[0] ," ", labels[0])

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100*correct/total))
    print("correct amount = ", correct, "\ntotal amount = ", total)

    log_event(0, "test of NN completed")
    return 100*(correct/total)


def main() -> None:

    writer = SummaryWriter("runs/cifar")
    if (input("would you like to load a model? (Y/N): ") == "N"):
        # cuda training check
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device in use: {device}")

        # get the data and train the model
        data: DataSet = get_shuffled_sets()
        my_nn = train_nn(data.train, device, data.test, writer)

        print("Training Finished")

        # this is where we test the model

        test_nn(data.test, my_nn)

        # We are saving the network weights
        print("Saving Model")
        torch.save(my_nn.state_dict(), 'network_weights.pth')
    else:
        # We are loading the network weights
        print("Loading NN")
        my_nn = NeuralNetwork()
        my_nn.load_state_dict(torch.load('network_weights.pth'))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        my_nn.to(device)
        print("Network weights loaded")

        # this is where we test the model
        data: DataSet = get_shuffled_sets()
        test_nn(data.test, my_nn)
    writer.close()


if __name__ == "__main__":

    liblog.start_logging()

    log: Logger = logging.getLogger(__name__)
    log_event(0, "started execution of main module")

    try:
        main()
    except KeyboardInterrupt:
        if liblog.debug_mode:
            log.info("got keyboard interrupt, exiting", exc_info=True)
        else:
            log.info("got keyboard interrupt, exiting")
        print("[info] got keyboard interrupt, exiting")
