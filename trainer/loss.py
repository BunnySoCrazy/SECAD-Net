import torch.nn as nn


class reconLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, outputs, gt_3d_occ):
        output_3d_occ = outputs["output_3d_occ"]
        total_2d_occ = outputs["total_2d_occ"]
        transformed_points = outputs["transformed_points"]
        h = outputs["h"]
        
        msk = (transformed_points[...,2]>-h).int() & (transformed_points[...,2]<=h).int()
        msk = msk.int()
        
        loss_recon = nn.MSELoss()(gt_3d_occ, output_3d_occ)
        loss_sketch = nn.MSELoss()(gt_3d_occ.unsqueeze(-1).repeat(1,1,total_2d_occ.shape[-1])*msk,
                                   total_2d_occ*msk)

        loss_recon = self.weights["recon_weight"] * loss_recon
        loss_sketch = self.weights["sketch_weight"] * loss_sketch
                     
        res = {"L_recon": loss_recon, 
               "L_sk": loss_sketch}
        return res