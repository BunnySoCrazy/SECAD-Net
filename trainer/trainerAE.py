import torch
from .base import BaseTrainer
from .loss import reconLoss
from .acc_recall import acc_recall
from model import Encoder,Decoder,Generator
from collections import OrderedDict


class TrainerAE(BaseTrainer):
    """Trainer for training SECAD-Net.
    """
    def build_net(self):
        self.encoder = Encoder().cuda()
        self.decoder = Decoder(num_primitives=self.specs["NumPrimitives"]).cuda()
        self.generator = Generator(num_primitives=self.specs["NumPrimitives"]).cuda()

    def set_optimizer(self, lr, betas):
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": lr, "betas": (betas[0], betas[1])},
                {"params": self.decoder.parameters(), "lr": lr, "betas": (betas[0], betas[1])},
                {"params": self.generator.parameters(), "lr": lr, "betas": (betas[0], betas[1])}
            ]
        )

    def set_loss_function(self):
        self.loss_func = reconLoss(self.specs["LossWeightTrain"]).cuda()

    def set_accuracy_function(self):
        self.acc_func = acc_recall().cuda()
        
    def forward(self, data):
        voxels = data['voxels'].cuda()
        occ_data = data['occ_data'].cuda()
        load_point_batch_size = occ_data.shape[1]
        point_batch_size = 16*16*16*2
        point_batch_num = int(load_point_batch_size/point_batch_size)
        which_batch = torch.randint(point_batch_num+1, (1,))
        
        if which_batch == point_batch_num:
            xyz = occ_data[:,-point_batch_size:, :3]
            gt_3d_occ = occ_data[:,-point_batch_size:, 3]
        else:
            xyz = occ_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, :3]
            gt_3d_occ = occ_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, 3]

        shape_code = self.encoder(voxels)
        shape_3d = self.decoder(shape_code)
        output_3d_occ,total_2d_occ,transformed_points = self.generator(xyz, shape_3d, shape_code)
        h = shape_3d[:,7,:].unsqueeze(1)

        outputs = {"output_3d_occ": output_3d_occ,
                   "total_2d_occ": total_2d_occ,
                   "transformed_points": transformed_points,
                   "h": h}
        loss_dict = self.loss_func(outputs, gt_3d_occ)
        acc_dict = self.acc_func(outputs, gt_3d_occ)
        
        return outputs, loss_dict, acc_dict
    
    def train_func(self, data):
        """one step of training"""
        self.encoder.train()
        self.decoder.train()
        self.generator.train()
        self.optimizer.zero_grad()
        outputs, losses, acc_recall = self.forward(data)
        self.update_network(losses)
        
        self.update_epoch_info(losses, acc_recall)
        if self.clock.step % 10 == 0:
            self.record_to_tb(losses, acc_recall)
        
        loss_info = OrderedDict({k: "{:.3f}".format(v.item()/(self.clock.minibatch+1))
                                for k, v in self.epoch_loss.items()})
        acc_info = OrderedDict({k: "{:.2f}".format(v.item()/(self.clock.minibatch+1))
                                for k, v in self.epoch_acc.items()})
        out_info = loss_info.copy()
        out_info.update(acc_info)
        return outputs, out_info