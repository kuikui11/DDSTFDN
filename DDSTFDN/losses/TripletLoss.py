import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper

class FullTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, feature, label):
        """
        N: batch size
        M: number of independent body parts
        C: feature dimention
        """
        #n,c,p -> p,n,c
        feature = feature.permute(2,0,1).contiguous()
        # [N, M, C] -> [M, N, C]
        # feature = feature.permute(1, 0, 2).contiguous()
        # [M, N]
        label = label.unsqueeze(0).repeat(feature.size(0), 1)

        M, N, C = feature.size()
        # [M, N, N]
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)

        # [M, N, N]
        dist = self.batch_dist(feature)
        dist = dist.view(-1)

        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(M, N, -1, 1)
        full_hn_dist = torch.masked_select(dist, hn_mask).view(M, N, 1, -1)
        full_loss_metric = F.relu(self.margin + full_hp_dist -
                                  full_hn_dist).view(M, -1)

        full_loss_metric_sum = full_loss_metric.sum(1)
        full_loss_num = (full_loss_metric != 0).sum(1).float()

        full_loss_metric_mean = full_loss_metric_sum / full_loss_num
        full_loss_metric_mean[full_loss_num == 0] = 0

        return full_loss_metric_mean.mean(
        ), full_loss_num.mean() / full_loss_metric.numel()

    def batch_dist(self, x):
        x2 = torch.sum(x**2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(
            1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist


class TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        # print(loss)
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)
        loss_avg = loss_avg.mean()
        # print(loss_avg)
        # print(loss_num)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        # print("loss_sum----")
        # print(loss_num)
        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist