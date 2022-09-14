import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Block(nn.Module):
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()

        group_width = cardinality * bottleneck_width        
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(self.expansion * group_width)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10, input_size=[32, 32]):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.layer4 = self._make_layer(num_blocks[3], 2)

        self.dropout = nn.Dropout(p = 0.5)
        self.linear1 = nn.Linear(cardinality * bottleneck_width * int(max(input_size) * (3/4)), cardinality * bottleneck_width * 8)
        self.bn1d = nn.BatchNorm1d(num_features=cardinality * bottleneck_width * 8)
        self.linear2 = nn.Linear(cardinality * bottleneck_width * int(max(input_size) /2), cardinality * bottleneck_width * 8)
        self.linear3 = nn.Linear(cardinality * bottleneck_width * 8 * int(max(input_size)/32 * (2/3)), num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        # out = F.relu(self.linear1(out))
        # out = self.bn1d(F.relu(self.linear2(out)))
        # out = F.softmax(self.linear3(out), dim=0)
        # out = F.softmax(self.linear3(out), dim=1)
        out = self.linear3(out)
        
        return out

class VGG16(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(VGG16, self).__init__()
        
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.block_3 = nn.Sequential(        
            nn.Conv2d(128, 256, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1),
            nn.ReLU(),        
            nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
          
        self.block_4 = nn.Sequential(   
            nn.Conv2d(256, 512, (3, 3), (1, 1), padding=1),
            nn.ReLU(),        
            nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1),
            nn.ReLU(),        
            nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.block_5 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), (1, 1),padding=1),
            nn.ReLU(),            
            nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1),
            nn.ReLU(),            
            nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1),
            nn.ReLU(),    
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))             
        )
            
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
            
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()
                    
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)

        return logits, probas

def ResNeXt29_2x64d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=2, bottleneck_width=64)

def ResNeXt29_4x64d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=4, bottleneck_width=64)
    
def ResNeXt29_8x64d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=8, bottleneck_width=64)
    
def ResNeXt29_32x4d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=32, bottleneck_width=4)

def test_resnext():
    net = ResNeXt29_32x4d()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())
    summary(net.cuda(), (3, 32, 32))

if __name__ == '__main__':
    test_resnext()
    