import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = nn.AvgPool2d((64, 64))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=(64, 64), align_corners=False,
                           mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat


class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'elevation_azimuth', 'range_azimuth', 'doppler_azimuth', 'elevation_range' and 'elevation_doppler'
    """

    def __init__(self, signal_type):
        super().__init__()
        self.signal_type = signal_type
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=128, k_size=3,
                                                      pad=(0, 1, 1), dil=1)
        self.azi_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.ele_max_pool = nn.MaxPool2d(2, stride=(1, 2))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                  pad=1, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1,
                                                pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)  # remove temporal dimension

        if self.signal_type in ('range_azimuth', 'doppler_azimuth'):
            # The Doppler dimension requires a specific processing
            x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0)
            x1 = self.azi_max_pool(x1_pad)
        
        elif self.signal_type in ('elevation_range', 'elevation_doppler'):
            x1_pad = F.pad(x1, (0, 0, 0, 1), "constant", 0)
            x1 = self.ele_max_pool(x1_pad)

        x2 = self.double_conv_block2(x1)
        x2_down = self.max_pool(x2)

        x3 = self.single_conv_block1_1x1(x2_down)
        # return input of ASPP block + latent features
        return x2_down, x3

class TMVA4D(nn.Module):
    """ 
    Temporal Multi-View network with ASPP for 4D radar data (TMVA4D)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Backbone (encoding)
        self.ea_encoding_branch = EncodingBranch('elevation_azimuth')
        self.da_encoding_branch = EncodingBranch('doppler_azimuth')
        self.ed_encoding_branch = EncodingBranch('elevation_doppler')
        self.er_encoding_branch = EncodingBranch('elevation_range')
        self.ra_encoding_branch = EncodingBranch('range_azimuth')

        # ASPP Blocks
        self.ea_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.da_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ed_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.er_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ra_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ea_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.da_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ed_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.er_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        # Decoding
        self.ea_single_conv_block2_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ea_upconv1 = nn.ConvTranspose2d(768, 128, 2, stride=2)
        self.ea_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ea_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.ea_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)

        # Final 1D convs
        self.ea_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)


    def forward(self, x_ea, x_da, x_ed, x_er, x_ra):
        # Backbone
        ea_features, ea_latent = self.ea_encoding_branch(x_ea)
        da_features, da_latent = self.da_encoding_branch(x_da)
        ed_features, ed_latent = self.ed_encoding_branch(x_ed)
        er_features, er_latent = self.er_encoding_branch(x_er)
        ra_features, ra_latent = self.ra_encoding_branch(x_ra)

        # ASPP blocks
        x1_ea = self.ea_aspp_block(ea_features)
        x1_da = self.da_aspp_block(da_features)
        x1_ed = self.ed_aspp_block(ed_features)
        x1_er = self.er_aspp_block(er_features)
        x1_ra = self.ra_aspp_block(ra_features)
        x2_ea = self.ea_single_conv_block1_1x1(x1_ea)
        x2_da = self.da_single_conv_block1_1x1(x1_da)
        x2_ed = self.ed_single_conv_block1_1x1(x1_ed)
        x2_er = self.er_single_conv_block1_1x1(x1_er)
        x2_ra = self.ra_single_conv_block1_1x1(x1_ra)

        # Latent Space
        # Features join either the RD or the RA branch
        x3 = torch.cat((ea_latent, da_latent, ed_latent, er_latent, ra_latent), 1)
        x3_ea = self.ea_single_conv_block2_1x1(x3)

        # Latent Space + ASPP features
        x4_ea = torch.cat((x2_ea, x3_ea, x2_da, x2_ed, x2_er, x2_ra), 1)

        # Parallel decoding branches with upconvs
        x5_ea = self.ea_upconv1(x4_ea)
        x6_ea = self.ea_double_conv_block1(x5_ea)
        x7_ea = self.ea_double_conv_block2(x6_ea)

        # Final 1D convolutions
        x8_ea = self.ea_final(x7_ea)

        return x8_ea