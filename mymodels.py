import torch
import torch.nn as nn
import os

__all__ = [
    'Color2Sketch', 'Sketch2Color', 'Discriminator', 
]

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)

class Conv2d_WS(nn.Conv2d):
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_chan, out_chan, kernel_size, stride, padding, dilation, groups, bias)
    
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1,1,1,1)+1e-5
        weight = weight / std.expand_as(weight)
        return torch.nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sample=None):
        super(ResidualBlock, self).__init__()
        self.ic = in_channels
        self.oc = out_channels
        self.conv1 = Conv2d_WS(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.conv2 = Conv2d_WS(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, out_channels)
        self.convr = Conv2d_WS(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnr = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sample = sample
        if self.sample == 'down':
            self.sampling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif self.sample == 'up':
            self.sampling = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        if self.ic != self.oc:
            residual = self.convr(x)
            residual = self.bnr(residual)
        else:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.sample is not None:
            out = self.sampling(out)
        return out

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            Conv2d_WS(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.GroupNorm(32, F_int)
            )
        
        self.W_x = nn.Sequential(
            Conv2d_WS(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.GroupNorm(32, F_int)
        )

        self.psi = nn.Sequential(
            Conv2d_WS(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi    
    
class Color2Sketch(nn.Module):
    def __init__(self, nc=3, pretrained=False):
        super(Color2Sketch, self).__init__()
        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                # Build ResNet and change first conv layer to accept single-channel input
                self.layer1 = ResidualBlock(nc, 64, sample='down')
                self.layer2 = ResidualBlock(64, 128, sample='down')
                self.layer3 = ResidualBlock(128, 256, sample='down')
                self.layer4 = ResidualBlock(256, 512, sample='down')
                self.layer5 = ResidualBlock(512, 512, sample='down')
                self.layer6 = ResidualBlock(512, 512, sample='down')
                self.layer7 = ResidualBlock(512, 512, sample='down')
                
            def forward(self, input_image):
                # Pass input through ResNet-gray to extract features
                x0 = input_image # nc * 256 * 256 
                x1 = self.layer1(x0) # 64 * 128 * 128 
                x2 = self.layer2(x1) # 128 * 64 * 64
                x3 = self.layer3(x2) # 256 * 32 * 32 
                x4 = self.layer4(x3) # 512 * 16 * 16 
                x5 = self.layer5(x4) # 512 * 8 * 8 
                x6 = self.layer6(x5) # 512 * 4 * 4
                x7 = self.layer7(x6) # 512 * 2 * 2

                return x1, x2, x3, x4, x5, x6, x7

        class Decoder(nn.Module):
            def __init__(self):
                super(Decoder, self).__init__()
                # Convolutional layers and upsampling     
                self.noise7 = ApplyNoise(512)
                self.layer7_up = ResidualBlock(512, 512, sample='up')
                
                self.Att6 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer6 = ResidualBlock(1024, 512, sample=None)
                self.noise6 = ApplyNoise(512)
                self.layer6_up = ResidualBlock(512, 512, sample='up')
                
                self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer5 = ResidualBlock(1024, 512, sample=None)
                self.noise5 = ApplyNoise(512)
                self.layer5_up = ResidualBlock(512, 512, sample='up')
                
                self.Att4 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer4 = ResidualBlock(1024, 512, sample=None)
                self.noise4 = ApplyNoise(512)
                self.layer4_up = ResidualBlock(512, 256, sample='up')
                
                self.Att3 = Attention_block(F_g=256,F_l=256,F_int=128)
                self.layer3 = ResidualBlock(512, 256, sample=None)
                self.noise3 = ApplyNoise(256)
                self.layer3_up = ResidualBlock(256, 128, sample='up')
                
                self.Att2 = Attention_block(F_g=128,F_l=128,F_int=64)
                self.layer2 = ResidualBlock(256, 128, sample=None)
                self.noise2 = ApplyNoise(128)
                self.layer2_up = ResidualBlock(128, 64, sample='up')
                
                self.Att1 = Attention_block(F_g=64,F_l=64,F_int=32)
                self.layer1 = ResidualBlock(128, 64, sample=None)
                self.noise1 = ApplyNoise(64)
                self.layer1_up = ResidualBlock(64, 32, sample='up')   
                
                self.noise0 = ApplyNoise(32)
                self.layer0 = Conv2d_WS(32, 3, kernel_size=3, stride=1, padding=1)
                self.activation = nn.ReLU(inplace=True)
                self.tanh = nn.Tanh()

            def forward(self, midlevel_input): #, global_input):
                x1, x2, x3, x4, x5, x6, x7 = midlevel_input
                
                x = self.noise7(x7)                
                x = self.layer7_up(x) # 512 * 4 * 4

                x6 = self.Att6(g=x,x=x6)
                x = torch.cat((x, x6), dim=1) # 1024 * 4 * 4
                x = self.layer6(x) # 512 * 4 * 4
                x = self.noise6(x)
                x = self.layer6_up(x) # 512 * 8 * 8 
                
                x5 = self.Att5(g=x,x=x5)
                x = torch.cat((x, x5), dim=1) # 1024 * 8 * 8
                x = self.layer5(x) # 512 * 8 * 8
                x = self.noise5(x)
                x = self.layer5_up(x) # 512 * 16 * 16 

                x4 = self.Att4(g=x,x=x4)
                x = torch.cat((x, x4), dim=1) # 1024 * 16 * 16
                x = self.layer4(x) # 512 * 16 * 16
                x = self.noise4(x)
                x = self.layer4_up(x) # 256 * 32 * 32 
                
                x3 = self.Att3(g=x,x=x3)
                x = torch.cat((x, x3), dim=1) # 512 * 32 * 32
                x = self.layer3(x) # 256 * 32 * 32 
                x = self.noise3(x)
                x = self.layer3_up(x) # 128 * 64 * 64 
                
                x2 = self.Att2(g=x,x=x2)
                x = torch.cat((x, x2), dim=1) # 256 * 64 * 64 
                x = self.layer2(x) # 128 * 64 * 64 
                x = self.noise2(x)
                x = self.layer2_up(x) # 64 * 128 * 128 
                
                x1 = self.Att1(g=x,x=x1)
                x = torch.cat((x, x1), dim=1) # 128 * 128 * 128
                x = self.layer1(x) # 64 * 128 * 128 
                x = self.noise1(x)
                x = self.layer1_up(x) # 32 * 256 * 256
                
                x = self.noise0(x)
                x = self.layer0(x) # 3 * 256 * 256
                x = self.tanh(x)

                return x

        self.encoder = Encoder()
        self.decoder = Decoder()
        if pretrained:
            print('Loading pretrained {0} model...'.format('Color2Sketch'), end=' ')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/color2edge/ckpt.pth')
            self.load_state_dict(checkpoint['netG'], strict=True)
            print("Done!")
        else:
            self.apply(weights_init)
            print('Weights of {0} model are initialized'.format('Color2Sketch'))
            
    def forward(self, inputs):
        encode = self.encoder(inputs)
        output = self.decoder(encode)
        
        return output
    
class Sketch2Color(nn.Module):
    def __init__(self, nc=3, pretrained=False):
        super(Sketch2Color, self).__init__()
        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                # Build ResNet and change first conv layer to accept single-channel input
                self.layer1 = ResidualBlock(nc, 64, sample='down')
                self.layer2 = ResidualBlock(64, 128, sample='down')
                self.layer3 = ResidualBlock(128, 256, sample='down')
                self.layer4 = ResidualBlock(256, 512, sample='down')
                self.layer5 = ResidualBlock(512, 512, sample='down')
                self.layer6 = ResidualBlock(512, 512, sample='down')
                self.layer7 = ResidualBlock(512, 512, sample='down')
                
            def forward(self, input_image):
                # Pass input through ResNet-gray to extract features
                x0 = input_image # nc * 256 * 256 
                x1 = self.layer1(x0) # 64 * 128 * 128 
                x2 = self.layer2(x1) # 128 * 64 * 64
                x3 = self.layer3(x2) # 256 * 32 * 32 
                x4 = self.layer4(x3) # 512 * 16 * 16 
                x5 = self.layer5(x4) # 512 * 8 * 8 
                x6 = self.layer6(x5) # 512 * 4 * 4
                x7 = self.layer7(x6) # 512 * 2 * 2

                return x1, x2, x3, x4, x5, x6, x7

        class Decoder(nn.Module):
            def __init__(self):
                super(Decoder, self).__init__()
                # Convolutional layers and upsampling     
                self.noise7 = ApplyNoise(512)
                self.layer7_up = ResidualBlock(512, 512, sample='up')
                
                self.Att6 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer6 = ResidualBlock(1024, 512, sample=None)
                self.noise6 = ApplyNoise(512)
                self.layer6_up = ResidualBlock(512, 512, sample='up')
                
                self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer5 = ResidualBlock(1024, 512, sample=None)
                self.noise5 = ApplyNoise(512)
                self.layer5_up = ResidualBlock(512, 512, sample='up')
                
                self.Att4 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer4 = ResidualBlock(1024, 512, sample=None)
                self.noise4 = ApplyNoise(512)
                self.layer4_up = ResidualBlock(512, 256, sample='up')
                
                self.Att3 = Attention_block(F_g=256,F_l=256,F_int=128)
                self.layer3 = ResidualBlock(512, 256, sample=None)
                self.noise3 = ApplyNoise(256)
                self.layer3_up = ResidualBlock(256, 128, sample='up')
                
                self.Att2 = Attention_block(F_g=128,F_l=128,F_int=64)
                self.layer2 = ResidualBlock(256, 128, sample=None)
                self.noise2 = ApplyNoise(128)
                self.layer2_up = ResidualBlock(128, 64, sample='up')
                
                self.Att1 = Attention_block(F_g=64,F_l=64,F_int=32)
                self.layer1 = ResidualBlock(128, 64, sample=None)
                self.noise1 = ApplyNoise(64)
                self.layer1_up = ResidualBlock(64, 32, sample='up')   
                
                self.noise0 = ApplyNoise(32)
                self.layer0 = Conv2d_WS(32, 3, kernel_size=3, stride=1, padding=1)
                self.activation = nn.ReLU(inplace=True)
                self.tanh = nn.Tanh()

            def forward(self, midlevel_input): #, global_input):
                x1, x2, x3, x4, x5, x6, x7 = midlevel_input
                
                x = self.noise7(x7)                
                x = self.layer7_up(x) # 512 * 4 * 4

                x6 = self.Att6(g=x,x=x6)
                x = torch.cat((x, x6), dim=1) # 1024 * 4 * 4
                x = self.layer6(x) # 512 * 4 * 4
                x = self.noise6(x)
                x = self.layer6_up(x) # 512 * 8 * 8 
                
                x5 = self.Att5(g=x,x=x5)
                x = torch.cat((x, x5), dim=1) # 1024 * 8 * 8
                x = self.layer5(x) # 512 * 8 * 8
                x = self.noise5(x)
                x = self.layer5_up(x) # 512 * 16 * 16 

                x4 = self.Att4(g=x,x=x4)
                x = torch.cat((x, x4), dim=1) # 1024 * 16 * 16
                x = self.layer4(x) # 512 * 16 * 16
                x = self.noise4(x)
                x = self.layer4_up(x) # 256 * 32 * 32 
                
                x3 = self.Att3(g=x,x=x3)
                x = torch.cat((x, x3), dim=1) # 512 * 32 * 32
                x = self.layer3(x) # 256 * 32 * 32 
                x = self.noise3(x)
                x = self.layer3_up(x) # 128 * 64 * 64 
                
                x2 = self.Att2(g=x,x=x2)
                x = torch.cat((x, x2), dim=1) # 256 * 64 * 64 
                x = self.layer2(x) # 128 * 64 * 64 
                x = self.noise2(x)
                x = self.layer2_up(x) # 64 * 128 * 128 
                
                x1 = self.Att1(g=x,x=x1)
                x = torch.cat((x, x1), dim=1) # 128 * 128 * 128
                x = self.layer1(x) # 64 * 128 * 128 
                x = self.noise1(x)
                x = self.layer1_up(x) # 32 * 256 * 256
                
                x = self.noise0(x)
                x = self.layer0(x) # 3 * 256 * 256
                x = self.tanh(x)

                return x

        self.encoder = Encoder()
        self.decoder = Decoder()
        if pretrained:
            print('Loading pretrained {0} model...'.format('Sketch2Color'), end=' ')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/edge2color/ckpt.pth')
            self.load_state_dict(checkpoint['netG'], strict=True)
            print("Done!")
        else:
            self.apply(weights_init)
            print('Weights of {0} model are initialized'.format('Sketch2Color'))
            
    def forward(self, inputs):
        encode = self.encoder(inputs)
        output = self.decoder(encode)
        
        return output
    
class Discriminator(nn.Module):
    def __init__(self, nc=6, pretrained=False):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.utils.spectral_norm(nn.Conv2d(nc, 64, kernel_size=4, stride=2, padding=1))
        self.bn1 = nn.GroupNorm(32, 64)
        self.conv2 = torch.nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.bn2 = nn.GroupNorm(32,128)
        self.conv3 = torch.nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.bn3 = nn.GroupNorm(32, 256)
        self.conv4 = torch.nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1))
        self.bn4 = nn.GroupNorm(32, 512)
        self.conv5 = torch.nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))               
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        if pretrained:
            pass
        else:
            self.apply(weights_init)
            print('Weights of {0} model are initialized'.format('Discriminator'))

    def forward(self, base, unknown):
        input = torch.cat((base, unknown), dim=1)
        x = self.activation(self.conv1(input))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))

        return x.mean((2,3))

# To initialize model weights
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('Conv2d_WS') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    elif classname.find('GroupNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    else:
        pass