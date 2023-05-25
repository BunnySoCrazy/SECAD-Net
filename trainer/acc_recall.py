import torch
import torch.nn as nn


class acc_recall(nn.Module):
    """This part of the code does not participate in backpropagation
       and is only used to evaluate the network predictions.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, gt_3d_occ):
        output_3d_occ = outputs["output_3d_occ"]

        predict_occupancies = (output_3d_occ > 0.5).float()
        target_occupancies = (gt_3d_occ > 0.5).float()
        
        accuracy = torch.sum(predict_occupancies*target_occupancies)/torch.sum(target_occupancies)
        recall = torch.sum(predict_occupancies*target_occupancies)/(torch.sum(predict_occupancies)+1e-9)
                     
        res = {"acc": accuracy, 
               "recall": recall}
        return res

