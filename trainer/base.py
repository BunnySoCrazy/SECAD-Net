import os
import torch
from abc import abstractmethod
from tensorboardX import SummaryWriter
from utils.workspace import get_model_params_dir,get_model_params_dir_shapename


class BaseTrainer(object):
    """Base trainer that provides common training behavior.
        All customized trainer should be subclass of this class.
    """
    def __init__(self, specs):
        self.specs = specs
        self.experiment_directory = specs['experiment_directory']
        self.log_dir = os.path.join(specs['experiment_directory'], 'log/')
        
        self.best_loss = float('inf')
        self.epoch_loss = None
        self.epoch_acc = None
        self.clock = TrainClock()

        # build network
        self.build_net()

        # set loss function
        self.set_loss_function()

        # set accuracy function
        self.set_accuracy_function()
        
        # set optimizer
        self.set_optimizer(lr=specs["LearningRate"],betas=specs["betas"])

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))

    @abstractmethod
    def build_net(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_optimizer(self):
        raise NotImplementedError
    
    def load_shape_code(self):
        """load shape code for finetuning"""
        pass
    
    def set_loss_function(self):
        """set loss function used in training"""
        pass
    
    def set_accuracy_function(self):
        """set accuracy function"""
        pass

    @abstractmethod
    def forward(self, data):
        """forward logic for your network"""
        """should return network outputs, losses(dict)"""
        raise NotImplementedError
    
    @abstractmethod
    def train_func(self, data):
        """one step of training"""
        raise NotImplementedError
    
    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        loss.backward()
        self.optimizer.step()

    def record_to_tb(self, loss_dict, acc_dict):
        """record loss to tensorboard"""
        losses_acc_values = {k: v.item() for k, v in loss_dict.items()|acc_dict.items()}
        tb = self.train_tb
        for k, v in losses_acc_values.items():
            tb.add_scalar(k, v, self.clock.step)
            
    def updata_epoch_info(self, loss_dict, acc_dict):
        if self.clock.minibatch == 0:
            self.epoch_loss = None
            self.epoch_acc = None
            
        self.epoch_loss = {key: self.epoch_loss.get(key, 0) + loss_dict[key] for key in loss_dict} \
                                                                if self.epoch_loss else loss_dict
        self.epoch_acc = {key: self.epoch_acc.get(key, 0) + acc_dict[key] for key in acc_dict} \
                                                                if self.epoch_acc else acc_dict
                                                                
    def save_model_parameters(self, filename):
        model_params_dir = get_model_params_dir(self.experiment_directory)
        torch.save(
            {"epoch": self.clock.epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "generator_state_dict": self.generator.state_dict(),
            "opt_state_dict": self.optimizer.state_dict()},
            os.path.join(model_params_dir, filename),
        )
        
    def save_model_if_best(self):
        epoch_loss_value = sum(self.epoch_loss.values()).item()/(self.clock.minibatch+1)
        if epoch_loss_value < self.best_loss:
            model_params_dir = get_model_params_dir(self.experiment_directory)
            torch.save(
                {"epoch": self.clock.epoch,
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "generator_state_dict": self.generator.state_dict(),
                "opt_state_dict": self.optimizer.state_dict()},
                os.path.join(model_params_dir, 'best.pth'),
            )
            self.best_loss = epoch_loss_value
            
    def save_model_parameters_per_shape(self, shapename, filename):

        model_params_dir = get_model_params_dir(self.experiment_directory)
        model_params_dir = get_model_params_dir_shapename(model_params_dir, shapename)

        torch.save(
            {"epoch": self.clock.epoch,
            "shape_code_state_dict": self.shape_code,
            "decoder_state_dict": self.decoder.state_dict(),
            "generator_state_dict": self.generator.state_dict(),
            "opt_state_dict": self.optimizer.state_dict()}, 
            os.path.join(model_params_dir, filename)
        )

    def load_model_parameters(self, checkpoint, opt=False):

        filename = os.path.join(
            self.experiment_directory, "ModelParameters", checkpoint + ".pth"
        )

        if not os.path.isfile(filename):
            raise Exception('model state dict "{}" does not exist'.format(filename))

        data = torch.load(filename)

        self.encoder.load_state_dict(data["encoder_state_dict"])
        self.decoder.load_state_dict(data["decoder_state_dict"])
        self.generator.load_state_dict(data["generator_state_dict"])
        if opt:
            self.optimizer.load_state_dict(data["opt_state_dict"])
        return data["epoch"]
    
    def load_model_parameters_per_shape(self, shapename, checkpoint):

        filename = os.path.join(
            self.experiment_directory, "ModelParameters", shapename, checkpoint + ".pth"
        )

        if not os.path.isfile(filename):
            raise Exception('model state dict "{}" does not exist'.format(filename))

        data = torch.load(filename)

        self.optimizer.load_state_dict(data["opt_state_dict"])
        self.decoder.load_state_dict(data["decoder_state_dict"])
        self.generator.load_state_dict(data["generator_state_dict"])
        
        return data["epoch"], data["shape_code_state_dict"]


class TrainClock(object):
    """ Clock object to track epoch and step during training
    """
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']
