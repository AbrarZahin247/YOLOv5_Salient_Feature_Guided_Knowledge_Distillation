import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM(nn.Module):
    def __init__(self, kernel_size=7,bias=False):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channels=channels
        self.r=r
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        # Reshape the pooled features to [batch_size, channels]
        avg_out = torch.flatten(avg_out, start_dim=1)
        max_out = torch.flatten(max_out, start_dim=1)

        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)

        out = avg_out + max_out
        return self.sigmoid(out)
        
    #     self.channels = channels
    #     self.r = r
    #     self.linear_max = nn.Sequential(
    #         nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    # def forward(self, x):
    #     max = F.adaptive_max_pool2d(x, output_size=1)
    #     avg = F.adaptive_avg_pool2d(x, output_size=1)
    #     b, c, _, _ = x.size()
    #     linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
    #     linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
    #     output = linear_max + linear_avg
    #     output = F.sigmoid(output) * x
    #     return output

class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM()
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        cam_out = self.cam(x)
        cam_out =cam_out.unsqueeze(-1).unsqueeze(-1)
        cam_out = x * cam_out
        # print(cam_out.size())
        sam_out=self.sam(x)
        # print(sam_out.size())
        tot_out=cam_out+sam_out
        # print(tot_out.size())
        return tot_out