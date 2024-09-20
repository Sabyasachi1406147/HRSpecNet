import torch
import torch.nn as nn
from torchsummary import summary

def set_fr_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    net = TF()
    if args.use_cuda:
        net.cuda() 
        print(net)
        summary(net, input_size=(2,6900))
    return net

class TF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        n_filters = 8
        signal_dim = 6900
        ae_filters = 64 
        kernel_size = 3 
        self.stride = 2
        self.output_padding = 1

        self.padding_conv = (((signal_dim//2)-1)*self.stride - signal_dim + (kernel_size-1) + 1)//2 + 1
        self.padding_trans = (2*signal_dim - (signal_dim-1)*self.stride - (kernel_size-1) - 1 - self.output_padding)//(-2)
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv1d(2,ae_filters,kernel_size,self.stride,self.padding_conv),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            nn.Conv1d(ae_filters,ae_filters,kernel_size,self.stride,self.padding_conv),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            nn.Conv1d(ae_filters,ae_filters,kernel_size,self.stride,self.padding_conv-1),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            nn.Conv1d(ae_filters,ae_filters,kernel_size,self.stride,self.padding_conv),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(ae_filters, ae_filters, kernel_size, self.stride, self.padding_trans, self.output_padding),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            nn.ConvTranspose1d(ae_filters, ae_filters, kernel_size, self.stride, self.padding_trans-1, self.output_padding-1),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            nn.ConvTranspose1d(ae_filters, ae_filters, kernel_size, self.stride, self.padding_trans, self.output_padding),
            nn.BatchNorm1d(ae_filters),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            nn.ConvTranspose1d(ae_filters, 2, kernel_size, self.stride, self.padding_trans, self.output_padding),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2),
            # nn.SELU(),
            )
        self.in_layer = nn.Sequential(
            nn.Conv1d(2, 256*n_filters, kernel_size=256, stride = 26, padding=0,
                                       bias=False),
            nn.BatchNorm1d(num_features=256*n_filters),
            )
        self.down1 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters*2,kernel_size,padding='same'),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.Conv2d(n_filters*2, n_filters*4,kernel_size,self.stride, self.padding_conv),
            nn.BatchNorm2d(n_filters*4),
            nn.ReLU(),
            )
        self.down3 = nn.Sequential(
            nn.Conv2d(n_filters*4, n_filters*8,kernel_size,self.stride,self.padding_conv),
            nn.BatchNorm2d(n_filters*8),
            nn.ReLU(),
            )
        self.down4 = nn.Sequential(
            nn.Conv2d(n_filters*8, n_filters*16,kernel_size,self.stride,self.padding_conv),
            nn.BatchNorm2d(n_filters*16),
            nn.ReLU(),
            )
        self.equal1 = nn.Sequential(
            nn.Conv2d(n_filters*16, n_filters*32,kernel_size,padding='same'),
            nn.BatchNorm2d(n_filters*32),
            nn.ReLU(),
            )
        self.equal2 = nn.Sequential(
            nn.Conv2d(n_filters*32, n_filters*16,kernel_size,padding='same'),
            nn.BatchNorm2d(n_filters*16),
            nn.ReLU(),
            )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(n_filters*16, n_filters*8,kernel_size,self.stride,self.padding_trans,self.output_padding),
            nn.BatchNorm2d(n_filters*8),
            nn.ReLU(),
            )
        self.cat1 = nn.Sequential(
            nn.ConvTranspose2d(n_filters*16, n_filters*8,kernel_size,padding=1),
            nn.BatchNorm2d(n_filters*8),
            nn.ReLU(),
            )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(n_filters*8, n_filters*4,kernel_size,self.stride,self.padding_trans,self.output_padding),
            nn.BatchNorm2d(n_filters*4),
            nn.ReLU(),
            )
        self.cat2 = nn.Sequential(
            nn.ConvTranspose2d(n_filters*8, n_filters*4,kernel_size,padding=1),
            nn.BatchNorm2d(n_filters*4),
            nn.ReLU(),
            )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(n_filters*4, n_filters*2,kernel_size,self.stride,self.padding_trans,self.output_padding),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(),
            )
        self.cat3 = nn.Sequential(
            nn.ConvTranspose2d(n_filters*4, n_filters*2,kernel_size,padding=1),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(),
            )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(n_filters*2, n_filters,kernel_size,padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            )
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, 5, stride=1,
                                            padding= 2, output_padding=0, bias=False)

        # self.rednet=REDNet30(self.n_layers,num_features=n_filters)
        # self.out_layer = nn.ConvTranspose2d(n_filters, 1, (3, 1), stride=(upsampling, 1),
        #                                     padding=(1, 0), output_padding=(1, 0), bias=False)
    
    def forward(self, x):
        bsz = x.size(0)
        encoded = self.encoder(x)
        x_ae = self.decoder(encoded)
        x = self.in_layer(x_ae)
        x = x.view(-1,8,256,256)
        x_reshape = x
        xd1 = self.down1(x)
        xd2 = self.down2(xd1)
        xd3 = self.down3(xd2)
        xd4 = self.down4(xd3)
        xe1 = self.equal1(xd4)
        xe2 = self.equal2(xe1)
        xu1 = self.up1(xe2)
        xcat1 = torch.cat((xu1,xd3),1)
        xcat1 = self.cat1(xcat1)
        xu2 = self.up2(xcat1)
        xcat2 = torch.cat((xu2,xd2),1)
        xcat2 = self.cat2(xcat2)
        xu3 = self.up3(xcat2)
        xcat3 = torch.cat((xu3,xd1),1)
        xcat3 = self.cat3(xcat3)
        xu4 = self.up4(xcat3)
        output_fr = self.out_layer(xu4)
        return x_ae,x_reshape,output_fr
    
