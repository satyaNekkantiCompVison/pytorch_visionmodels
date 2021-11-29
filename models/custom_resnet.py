import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic block of the ResNet.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        """
        Forward method.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out


## Assignment Based Custom network
class CustomResNetModule(nn.Module):
    """
    ResNet Architecture.
    """

    def __init__(self, block, num_classes=10):
        super().__init__()

        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.prepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),  
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Conv1Layer X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.Conv1Layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # ResBlock1( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        self.ResBlock1 = block(128, 128)

        # Conv2Layer X = Conv 3x3 (s1,p1) >> MaxPooling2D >> BN >> ReLU [256k]
        self.Conv2Layer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Conv3Layer X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        self.Conv3Layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # ResBlock2 ( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        self.ResBlock2 = block(512, 512)

        # MaxPooling with Kernel Size 4
        self.pooling = nn.MaxPool2d(4, 4)

        # Fully Connected Layer
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward method.
        """
        X = self.prepLayer(x)
        X = self.Conv1Layer(X)
        R1 = self.ResBlock1(X)
        X = X + R1
        X = self.Conv2Layer(X)
        X = self.Conv3Layer(X)
        R2 = self.ResBlock2(X)
        X = X + R2
        X = self.pooling(X)
        X = X.view(X.size(0), -1)
        X = self.fc1(X)
        return X


def CustomResNet():
    """
    Custom ResNet model.
    """
    return CustomResNetModule(BasicBlock)