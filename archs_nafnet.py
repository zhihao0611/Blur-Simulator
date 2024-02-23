import torch.nn as nn
import torch.nn.functional as F
from torch import cat

from .simulator import BlurDictModel, RegisterGridTime
from .layers_nafnet import LayerNorm2d, NAFBlock, UpNAFNet
from .laplacian_pyramid import LapPyramid, UpsampleGauss


# This is to calculate the blur kernels from pairs of sharp/blur images
# Should only be use during training of the pseudo inverse
# Make sure to fix the model by setting all paremeters to trainable=False
# Also, to train faster, use encapsule inside with torch.nograd()
class BlurSimul(nn.Module):
    def __init__(self, in_channels, nBase, kernel_size, width=32, enc_blk_nums=[1, 1], middle_blk_num=5, dec_blk_nums=[1, 1], bias=True):
        super().__init__()
        self.add_module('lap',     LapPyramid(max_levels=3, channels=in_channels))
        self.add_module('up_pos',  UpsampleGauss(2))
        
        chan  = width
        
        self.intro       = nn.Sequential(nn.Conv2d(in_channels=in_channels*2, out_channels=chan, kernel_size=3, padding=1, stride=1, groups=1, bias=True))
        self.encoders    = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders    = nn.ModuleList()
        self.downs       = nn.ModuleList()
        self.ups         = nn.ModuleList()
        

        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for i in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan  = chan * 2
                
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for i in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                UpNAFNet(chan, 2),
            )
            chan  = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for i in range(num)]
                )
            )
        
        self.out = nn.ModuleList([
            nn.Conv2d(width*4, nBase, 3, padding=1, bias=bias),
            nn.Conv2d(width*2, nBase, 3, padding=1, bias=bias),
            nn.Conv2d(width,   nBase, 3, padding=1, bias=bias)
        ])

        self.b3 = BlurDictModel(nBase, kernel_size)
        self.b2 = BlurDictModel(nBase, kernel_size)
        self.b1 = BlurDictModel(nBase, kernel_size)

        self.nBase       = nBase
        self.in_channels = in_channels
        self.padder_size = 2 ** len(self.encoders)
        
    def forward(self, inp):
        #Prepare images
        x, y = inp     
        B, C, H, W = y.shape # 32, 3, 128, 128
        B, C, H, W = x.shape
        y = self.check_image_size(y)
        x = self.check_image_size(x)

        #Network
        f = self.intro(cat([x, y], dim=1))

        for encoder, down in zip(self.encoders, self.downs):
            f = encoder(f)
            f = down(f)

        f = self.middle_blks(f)

        b = [self.out[0](f), ]

        for decoder, up, head in zip(self.decoders, self.ups, self.out[1: ]):
            f = up(f)
            f = decoder(f)
            b.append(head(f))
                
        #Blur (multi scale)
        # hi_y, lw_y = self.lap(y)
        hi_x, lw_x = self.lap(x)

        y3 = self.b3([lw_x[1], b[0]]) 
        y2 = self.b2([hi_x[1], b[1]]) + self.lap.up(y3)
        y1 = self.b1([hi_x[0], b[2]]) + self.lap.up(y2)         

        return [y1, y2, y3]
        #return [y, lw_y[0], lw_y[1]], [y1, y2, y3]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h)) # 32, 3, 128, 128
        return x

    def set_trainable_module(self, module_name, trainable):
        module = getattr(self, module_name)

        for p in module.parameters():
            p.requires_grad = trainable