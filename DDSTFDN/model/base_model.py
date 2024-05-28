import torch
from DenseGait_BaseOpenGait.model.modules import *
from DenseGait_BaseOpenGait.model.backbone import DenseGait3D,DenseGait3D_OUMVLP
from DenseGait_BaseOpenGait.utils.common import init_seeds,config_loader,get_attr_from,get_valid_args,get_ddp_module


class BaseModel(torch.nn.Module):
    def __init__(self,model_cfg,device):#
        super(BaseModel, self).__init__()
        self.Backbone = DenseGait3D()
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.MultiScaleConv1D = MultiScaleConv1D(input_channel=256, output_channel=256)
        self.MSSA = MSSA()

        self.device = device

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]
             if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        sils = sils.to(self.device)
        outs = self.Backbone(sils)  # [n, c, s, h, w]

       
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]


        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1
        return embed,logits