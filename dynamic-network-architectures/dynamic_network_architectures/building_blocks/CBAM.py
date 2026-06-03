from torch import nn
import torch

class ChannelAttention(nn.Module):
    def __init__(self, 
                channels: int, # Output channels 
                conv_op,
                reduction, 
                spatial_kernel):
        super().__init__()
        is_3d = conv_op == nn.Conv3d

        if is_3d:
            # -> (B, C, 1, 1, 1)
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_channels = max(channels // reduction, 1)

        # Channel attention
        self.mlp = nn.Sequential(
            conv_op(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            conv_op(hidden_channels, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid_channel(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, 
                channels: int, # Output channels 
                conv_op,
                reduction, 
                spatial_kernel):
        super().__init__()
        is_3d = conv_op == nn.Conv3d

        if is_3d:
            norm_op = nn.InstanceNorm3d
        else:
            norm_op = nn.InstanceNorm2d

        self.spatial = nn.Sequential(
            conv_op(2, 1, kernel_size=spatial_kernel,
                    padding=spatial_kernel // 2,
                    bias=False),
            norm_op(1)
        )

        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # -> (B, 1, D, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # -> (B, 1, D, H, W)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1)) # self.spatial(tensor(B, 2, D, H, W))
        return self.sigmoid_spatial(spatial_att)



class CBAM(nn.Module):
    def __init__(self, 
                 channels: int, # Output channels 
                 conv_op,

                 reduction=8, 
                 spatial_kernel=7):
        super().__init__()
        print("Initilized CBAM")
        self.channel_attention = ChannelAttention(channels, conv_op, reduction, spatial_kernel)
        self.spatial_attention = SpatialAttention(channels, conv_op, reduction, spatial_kernel)

    def forward(self, x):
        # Channel
        x = x * self.channel_attention(x)
        # Spatial
        x = x * self.spatial_attention(x)

        return x
    

if __name__ == "__main__":
    x = torch.rand((1, 3, 40, 32))

    cbam = CBAM(
        channels=3,
        conv_op=nn.Conv2d,
        reduction=8
    )

    out = cbam(x)
    with torch.no_grad:
        print("in", x.mean(), "Out", out.mean())
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)



    # is_3d = conv_op == nn.Conv3d

    #     if is_3d:
    #         # -> (B, C, 1, 1, 1)
    #         self.avg_pool = nn.AdaptiveAvgPool3d(1)
    #         self.max_pool = nn.AdaptiveMaxPool3d(1)
    #     else:
    #         self.avg_pool = nn.AdaptiveAvgPool2d(1)
    #         self.max_pool = nn.AdaptiveMaxPool2d(1)

    #     hidden_channels = max(channels // reduction, 1)

    #     # Channel attention
    #     self.mlp = nn.Sequential(
    #         conv_op(channels, hidden_channels, 1, bias=False),
    #         nn.ReLU(inplace=True),
    #         conv_op(hidden_channels, channels, 1, bias=False)
    #     )

    #     self.sigmoid_channel = nn.Sigmoid()

    #     # Spatial attention
    #     self.spatial = nn.Sequential(
    #         conv_op(2, 1, kernel_size=spatial_kernel,
    #                 padding=spatial_kernel // 2,
    #                 bias=False),
    #         norm_op(channels)
    #     )