import torch
import torch.nn as nn
from torchsummary import summary
# torch version = 1.2.0
class Mish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x*torch.tanh(nn.Softplus()(x))

class baseConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,mish=True) -> None:
        super().__init__()
        if mish:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                nn.BatchNorm2d(out_channels),
                Mish(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        return self.conv(x)

class resBlock(nn.Module):
    def __init__(self,in_channels,out_channels=None) -> None:
        super().__init__()
        if out_channels==None:
            out_channels = in_channels
        self.conv = nn.Sequential(
            baseConv(in_channels,out_channels,kernel_size=1,stride=1,padding=0),
            baseConv(out_channels,in_channels,kernel_size=3,stride=1,padding=1,mish=False),
        )
    def forward(self, x):
        return Mish()(x + self.conv(x))

class cspResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,num_blocks,first=False) -> None:
        super().__init__()
        if first:
            self.downsample = baseConv(in_channels,out_channels,kernel_size=3,stride=2,padding=1)
            self.route1 = baseConv(out_channels,out_channels,kernel_size=1,stride=1,padding=0)
            self.route2 = nn.Sequential(
                baseConv(out_channels,out_channels,kernel_size=1,stride=1,padding=0),
                resBlock(out_channels,out_channels//2),
            )
            self.con = baseConv(int(2*out_channels),out_channels,kernel_size=1,stride=1,padding=0)
        else:
            self.downsample = baseConv(in_channels,out_channels,kernel_size=3,stride=2,padding=1)
            self.route1 = baseConv(out_channels,out_channels//2,kernel_size=1,stride=1,padding=0)
            self.route2 = nn.Sequential(
                baseConv(out_channels,out_channels//2,kernel_size=1,stride=1,padding=0),
                nn.Sequential(*[resBlock(out_channels//2,out_channels//2) for _ in range(num_blocks)])
            )
            self.con = baseConv(out_channels,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        x = self.downsample(x)
        route1 = self.route1(x)
        route2 = self.route2(x)
        x = torch.cat([route1,route2],dim=1)
        x = self.con(x)
        return x

class CspDarkNet_53(nn.Module):
    def __init__(self,num_blocks,num_classes = 10) -> None:
        super().__init__()
        channels = [64,128,256,512,1024]
        self.conv1 = baseConv(3,32,kernel_size=3,stride=1,padding=1)
        self.neck = cspResBlock(32,channels[0],num_blocks=None,first=True)
        self.body = nn.Sequential(
            *[cspResBlock(channels[i],channels[i+1],num_blocks[i]) for i in range(4)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1],num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.neck(x)
        x = self.body(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = CspDarkNet_53([2,8,8,4])
    print(model)
    # if you have GPU, You can use the following code to show this net
    # device = torch.device('cuda:0')
    # model.to(device=device)
    # torchsummary.summary(model,input_size=(3,416,416))
    
