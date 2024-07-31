import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        ngf = 64
        # input：[100,1,1]
        self.g1 = nn.Sequential(
            nn.ConvTranspose2d(100, ngf*8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True)
        )
        # input：[ngf*8,4,4]
        self.g2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True)
        )
        # input：[ngf*4,8,8]
        self.g3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
        )
        # input：[ngf*2,16,16]
        self.g4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf*1, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf*1),
            nn.ReLU(True)
        )
        # input：[ngf,32,32]
        self.g5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )
        # output：[3,64,64]

    def forward(self, x):
        y = self.g1(x)
        y = self.g2(y)
        y = self.g3(y)
        y = self.g4(y)
        y = self.g5(y)
        return y

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        ndf = 64

        self.d1 = nn.Sequential(
            nn.Conv2d(3,ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.d3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.d4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.d5 = nn.Sequential(
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self,x):
        y = self.d1(x)
        y = self.d2(y)
        y = self.d3(y)
        y = self.d4(y)
        y = self.d5(y)
        y = y.view(-1,1)
        return y

def feature3_concat(feature1, feature2, feature3):
    c1, c2, c3 = feature1.size()[1], feature2.size()[1], feature3.size()[1]
    feature_total = torch.concat((feature1, feature2, feature3), dim=1)
    count = 0
    feature_total[:, count:count + c1 // 3, :, :] = feature1[:, 0:c1 // 3, :, :];
    count += c1 // 3
    feature_total[:, count:count + c2 // 3, :, :] = feature2[:, 0:c2 // 3, :, :];
    count += c2 // 3
    feature_total[:, count:count + c3 // 3, :, :] = feature3[:, 0:c3 // 3, :, :];
    count += c3 // 3

    feature_total[:, count:count + c1 // 3, :, :] = feature1[:, c1 // 3:2 * c1 // 3, :, :];
    count += c1 // 3
    feature_total[:, count:count + c2 // 3, :, :] = feature2[:, c2 // 3:2 * c2 // 3, :, :];
    count += c2 // 3
    feature_total[:, count:count + c3 // 3, :, :] = feature3[:, c3 // 3:2 * c3 // 3, :, :];
    count += c3 // 3

    feature_total[:, count:count + c1 // 3, :, :] = feature1[:, 2 * c1 // 3:, :, :];
    count += c1 // 3
    feature_total[:, count:count + c2 // 3, :, :] = feature2[:, 2 * c2 // 3:, :, :];
    count += c2 // 3
    feature_total[:, count:count + c3 // 3, :, :] = feature3[:, 2 * c3 // 3:, :, :];
    count += c3 // 3

    return feature_total

def feature2_concat(image, feature):
    num = feature.size()[1]
    count = 0
    x = torch.cat((image, feature), dim=1)
    x[:, count:count + 1, :, :] = image[:, 0:1, :, :]
    count += 1
    x[:, count:count + num // 3, :, :] = feature[:, :num // 3, :, :]
    count += num // 3
    x[:, count:count + 1, :, :] = image[:, 1:2, :, :]
    count += 1
    x[:, count:count + num // 3, :, :] = feature[:, num // 3:2 * num // 3, :, :]
    count += num // 3
    x[:, count:count + 1, :, :] = image[:, 2:3, :, :]
    count += 1
    x[:, count:, :, :] = feature[:, 2 * num // 3:, :, :]
    return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels=3):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=120, kernel_size=3, padding=1, groups=3)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=60, kernel_size=4, padding=1, groups=3)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=30, kernel_size=5, padding=2, groups=3)

        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = self.act(self.conv3(x))

        y = feature3_concat(x1, x2, x3)

        return y

class BasicBlock_WithoutAct(nn.Module):
    def __init__(self, in_channels=3):
        super(BasicBlock_WithoutAct, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=120, kernel_size=3, padding=1, groups=3)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=60, kernel_size=4, padding=1, groups=3)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=30, kernel_size=5, padding=2, groups=3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = self.conv3(x)

        y = feature3_concat(x1, x2, x3)

        return y

class ResBlock(nn.Module):
    def __init__(self, in_channels=3):
        super(ResBlock, self).__init__()

        self.step1 = BasicBlock(in_channels=in_channels)
        self.step2 = BasicBlock_WithoutAct(in_channels=210)

    def forward(self, x):
        y = self.step1(x)
        y = self.step2(y)
        return y

class ChannelAttention(nn.Module):
    def __init__(self, in_channels=210, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pooling = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc1 = nn.Linear(in_features=in_channels, out_features=in_channels // ratio, bias=False)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(in_features=in_channels // ratio, out_features=in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, w, h = x.shape

        max_pool = self.max_pooling(x).view([b, c])
        avg_pool = self.avg_pooling(x).view([b, c])

        y1 = self.relu(self.fc1(max_pool))
        y2 = self.relu(self.fc1(avg_pool))

        y1 = self.fc2(y1)
        y2 = self.fc2(y2)

        y = self.sigmoid(y1 + y2).view([b, c, 1, 1])
        y = x * y

        return y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding="same", bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_maxpool, _ = torch.max(x, dim=1, keepdim=True)
        x_avgpool = torch.mean(x, dim=1, keepdim=True)

        y = torch.concat([x_maxpool, x_avgpool], dim=1)
        y = self.sigmoid(self.conv(y))

        y = x * y

        return y

class CBAM(nn.Module):
    def __init__(self, in_channels=210, ratio=4, kernel_size=5):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels=in_channels, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        y = self.channel_attention(x)
        y = self.spatial_attention(y)
        return y

class ResWithCAMB(nn.Module):
    def __init__(self, in_channels=210, ratio=4, kernel_size=5):
        super(ResWithCAMB, self).__init__()

        self.res_block = ResBlock(in_channels=in_channels)
        self.attention = CBAM(in_channels=in_channels, ratio=ratio, kernel_size=kernel_size)

    def forward(self, x):
        y = self.res_block(x)
        y = self.attention(y)

        return y

class Hiding(nn.Module):
    def __init__(self):
        super(Hiding, self).__init__()

        ## PreparationNetwork
        self.pre1 = BasicBlock(in_channels=3)
        self.pre2 = BasicBlock(in_channels=210)
        ## hidingNetwork
        self.hide1 = BasicBlock(in_channels=213)

        self.hide2 = ResWithCAMB(in_channels=210)
        self.hide3 = ResWithCAMB(in_channels=210)

        # ending
        self.conv1 = nn.Conv2d(in_channels=210, out_channels=90, kernel_size=3, padding=1, groups=3)
        self.conv2 = nn.Conv2d(in_channels=90, out_channels=3, kernel_size=3, padding=1, groups=3)

    def forward(self, x, y):
        ## preparationNetwork
        y = self.pre1(y)
        y = self.pre2(y)

        ## HidingNetwork
        z = feature2_concat(x, y)

        z = self.hide1(z)

        sc = self.hide2(z)
        z = torch.relu(z + sc)

        sc = self.hide3(z)
        z = torch.relu(z + sc)

        z = torch.tanh(self.conv1(z))
        z = torch.tanh(self.conv2(z))

        return z

class Module_conceal(nn.Module):
    def __init__(self):
        super(Module_conceal, self).__init__()
        self.gnet = Generator()
        self.hnet = Hiding()

    def forward(self, fixed_noise,secret):
        pmap = self.gnet(fixed_noise)
        stego = self.hnet(pmap,secret)
        return stego

class Module_reveal(nn.Module):
    def __init__(self):
        super(Module_reveal, self).__init__()

        self.extract1 = BasicBlock(in_channels=3)

        self.extract2 = ResWithCAMB(in_channels=210)
        self.extract3 = ResWithCAMB(in_channels=210)
        self.extract4 = ResWithCAMB(in_channels=210)

        self.conv1 = nn.Conv2d(in_channels=210, out_channels=90, kernel_size=3, padding=1, groups=3)
        self.conv2 = nn.Conv2d(in_channels=90, out_channels=3, kernel_size=3, padding=1, groups=3)

    def forward(self, x):
        y = self.extract1(x)

        sc = self.extract2(y)
        y = torch.relu(y + sc)

        sc = self.extract3(y)
        y = torch.relu(y + sc)

        sc = self.extract4(y)
        y = torch.relu(y + sc)

        y = torch.tanh(self.conv1(y))
        y = torch.tanh(self.conv2(y))

        return y

