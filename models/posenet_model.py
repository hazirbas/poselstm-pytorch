import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pickle
import numpy

class PoseNetModel(BaseModel):
    def name(self):
        return 'PoseNetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc)

        # load/define networks
        googlenet_file = open("/remwork/atcremers72/hazirbas/projects/posenet_bashing/pretrained_models/places-googlenet/places-googlenet.pickle", "rb")
        googlenet_weights = pickle.load(googlenet_file, encoding="bytes")
        googlenet_file.close()
        self.mean_image = np.load(os.path.join(opt.dataroot , 'mean_image.npy'))

        self.netG = networks.define_G(opt.input_nc, None, None, opt.which_model_netG,
                                      init_from=googlenet_weights, isTest=not self.isTrain,
                                      gpu_ids = self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionXYZ = [torch.nn.MSELoss()] * 3
            self.criterionWPQR = [torch.nn.MSELoss()] * 3

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, eps=1,
                                                weight_decay=0.0625,
                                                betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            # for optimizer in self.optimizers:
            #     self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.image_paths = input['A_paths']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        self.loss_G = 0
        self.loss_aux = np.array([0, 0, 0, 0, 0])
        loss_weights = [self.opt.beta/0.3, self.opt.beta/0.3, self.opt.beta]
        for l, beta in enumerate(loss_weights):
            ## normalize rotation
            # norm = torch.norm(self.fake_B[2*l+1], p=2, dim=1, keepdim=True)
            mse_pos = self.criterionXYZ[l](self.fake_B[2*l], self.real_B[:, 0:3])
            mse_ori = self.criterionWPQR[l](self.fake_B[2*l+1], self.real_B[:, 3:]) * beta
            self.loss_G += mse_pos + mse_ori
            self.loss_aux[l] = self.loss_G.data[0]
            if l == 2:
                self.loss_aux[l+1] = mse_pos.data[0]
                self.loss_aux[l+2] = mse_ori.data[0]
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('G_aux1', self.loss_aux[0]),
                                ('G_aux2', self.loss_aux[1] - self.loss_aux[0]),
                                ('G_final', self.loss_aux[2] - self.loss_aux[1].sum()),
                                ('mse_pos_final', self.loss_aux[3]),
                                ('mse_ori_final', self.loss_aux[4]),
                                ])

        ori_distance = torch.acos(torch.abs(self.fake_B[1].mul(self.real_B[:, 3:]).sum()))
        return [torch.dist(self.fake_B[0], self.real_B[:, 0:3])[0].data[0],
                2*180/numpy.pi * ori_distance[0].data[0]]

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        # fake_B = util.tensor2im(self.fake_B.data)
        # real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
