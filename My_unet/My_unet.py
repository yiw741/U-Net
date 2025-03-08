import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding=1 以保持尺寸
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding=1 以保持尺寸
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=2, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channel, feature))
            in_channel = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2, padding=0))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channel, kernel_size=1)

    def cut(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # 上采样
            skip_connection = skip_connections.pop()  # 获取跳跃连接

            # 确保 skip_connection 和 x 的形状一致，长宽一致
            if skip_connection.shape[2:] != x.shape[2:]:
                # 如果形状不一致，则调整 x 的大小以匹配 skip_connection，使用的是双线性插值
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat = torch.cat((skip_connection, x), dim=1)  # 拼接，通道数
            x = self.ups[idx + 1](concat)  # 通过下一个卷积层

        return self.final_conv(x)

if __name__ == '__main__':
    input_data = torch.randn([1, 1, 572, 572])
    unet = UNet()
    output = unet(input_data)
    print(output.shape)  # 应该输出 torch.Size([1, 2, 388, 388])
