import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from DenseGait_BaseOpenGait.utils.common import clones, is_list_or_tuple




class HorizontalPoolingPyramid(nn.Module):

    def __init__(self, bin_num=None):
        super(HorizontalPoolingPyramid, self).__init__()
        if bin_num == 1:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p]
        """
        n, c, h, w = x.size()
        features = []
        for b in self.bin_num:
            # print(b)
            z_gap = F.adaptive_avg_pool2d(x, (b, 1))
            # print(z_gap.size())
            z_gmp = F.adaptive_max_pool2d(x, (b, 1))
            # print(z_gmp.size())
            z = torch.add(z_gap, z_gmp)
            z = z.view(n, c, -1)
            # print(z.size())
            features.append(z)
        return torch.cat(features, -1)


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)

class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()

class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.contiguous().view(n, -1)  # [n, c*p]
            # x = x.reshape(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()



class _P3DA(nn.Module):
    def __init__(self,num_input_features,growth_rate):
        """
        :param num_input_features:
        :param growth_rate:
        :param bn_size:
        :param drop_rate:
        """
        super(_P3DA, self).__init__()
        self.norm = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=num_input_features,out_channels=growth_rate,
                                          kernel_size=3,stride=1,padding=1,bias=False)
        # self.norm = nn.BatchNorm3d(growth_rate)
        # self.relu = nn.ReLU(inplace=True)

        self.conv2 = S_Conv(in_planes=num_input_features, out_planes=num_input_features,
                                           kernal_size=3, stride=(1,1,1), padding = (0,1,1))
        self.norm2 = nn.BatchNorm3d(num_input_features)
        self.conv3 = T_Conv(in_planes=num_input_features, out_planes=num_input_features,
                                           kernal_size=3, stride=(1,1,1), padding = (1,0,0))

        self.norm4 = nn.BatchNorm3d(num_input_features)
        self.conv4 = nn.Conv3d(in_channels=num_input_features, out_channels=growth_rate,
                               kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x0 = self.norm(x)
        x0 = self.relu(x0)
        x1 = self.conv1(x0)
        # x1 = self.norm(x1)
        # x1 = self.relu(x1)
        out = self.conv2(x0)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm4(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = torch.add(out,x1)
        # out = torch.cat([out, x1], 1)
        # out = out + x1
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self,channel,ratio=16):  
        super(ChannelAttentionModule, self).__init__()

  
        self.shared_MLP = nn.Sequential(
            nn.Conv3d(in_channels=channel,out_channels=channel // ratio,kernel_size=1,bias=False),
            nn.BatchNorm3d(channel // ratio),
            nn.ReLU(),
            nn.Conv3d(in_channels=channel // ratio,out_channels=channel,kernel_size=1,bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):

        out = self.sigmoid(self.shared_MLP(x))
        return out

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2,out_channels=1,kernel_size=7,stride=1,padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avgout = torch.mean(x,dim=1,keepdim=True)
        maxout,index_ = torch.max(x,dim=1,keepdim=True)
        out = torch.cat([avgout,maxout],dim=1) 
        out = self.sigmoid(self.conv3d(out))
        return out

class CBAM(nn.Module):
    def __init__(self,channel):
        super(CBAM, self).__init__()
        self.channel__attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self,x):
        # out = self.channel__attention(x) * x
        out = torch.mul(self.channel__attention(x),x)
        # CBAM_01 = out
        out = torch.mul(self.spatial_attention(out), out)
        return out


