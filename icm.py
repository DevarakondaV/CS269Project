# -*- coding: utf-8 -*-
"""ICM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14yXZBEs7dEN-Rqwa2J1STjhvIIT9fDmE
"""

import torch
import os
import torch.nn as nn
import numpy as np
import time
from tensorboardX import SummaryWriter
from torchvision import datasets as datasets_torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import importlib
import pandas as pd

torch_dtype = torch.float32
torch.set_default_dtype(torch_dtype)

# Model arguments
class Args:
  def __init__(self):
    self.input_size = 1
    self.seed = 11
    self.cuda = False
    self.name = ''
    self.dataset = 'mnist'
    self.num_experts = 3
    self.batch_size = 32
    self.learning_rate_initialize = 1E-2
    self.learning_rate_expert = 1E-3
    self.learning_rate_discriminator = 1E-3
    self.epochs_init = 25
    self.epochs = 20
    self.optimizer_initialize = 'adam'
    self.optimizer_experts = 'adam'
    self.optimizer_discriminator = 'adam'
    self.outdir = '/home/vishnu/Documents/EngProjs/Masters/HC/'
    self.datadir = '/home/vishnu/Documents/EngProjs/Masters/HC/data/'
    self.load_initialized_experts = True
    self.model_for_initialized_experts = 'mnist_n_exp_3_bs_32_lri_0.01_lre_0.001_lrd_0.001_ei_25_e_20_oi_adam_oe_adam_oe_adam_1653980272'
    self.inference_experts = True
    self.model_for_trained_experts = self.model_for_initialized_experts
    self.device =  torch.device("cuda" if self.cuda else "cpu")
    self.weight_decay = 0
    self.log_interval = 10
    self.iterations = 3000

args = Args()

torch.manual_seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# Experiment name
timestamp = str(int(time.time()))
if args.name == '':
    name = '{}_n_exp_{}_bs_{}_lri_{}_lre_{}_lrd_{}_ei_{}_e_{}_oi_{}_oe_{}_oe_{}_{}'.format(
        args.dataset, args.num_experts, args.batch_size, args.learning_rate_initialize,
        args.learning_rate_expert, args.learning_rate_discriminator, args.epochs_init,
        args.epochs, args.optimizer_initialize, args.optimizer_experts, args.optimizer_discriminator,
        timestamp)
    args.name = name
else:
    args.name = '{}_{}'.format(args.name, timestamp)
print('\nExperiment: {}\n'.format(args.name))

# Logging. To run: tensorboard --logdir <args.outdir>/logs
log_dir = os.path.join(args.outdir, 'logs')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_dir_exp = os.path.join(log_dir, args.name)
os.mkdir(log_dir_exp)
writer = SummaryWriter(log_dir=log_dir_exp)

# Directory for checkpoints
checkpt_dir = os.path.join(args.outdir, 'checkpoints')
if not os.path.exists(checkpt_dir):
    os.mkdir(checkpt_dir)

class MNISTDataset(Dataset):
    def __init__(self, args):
        self.before_path = os.path.join(args.datadir, "mnist_before6.npy")
        self.after_path = os.path.join(args.datadir, "mnist_after6.npy")
        # self.before_data = pd.read_csv(self.before_path, skiprows = 2)
        # self.after_data = pd.read_csv(self.after_path, skiprows = 2)
        self.before_data = np.load(self.before_path)[:10000]
        self.after_data = np.load(self.after_path)[:10000]
        self.args = args
        self.im_size = int(self.before_data.size/len(self.before_data))

    def __getitem__(self, index):
        # x_before = np.array(self.before_data.iloc[index])[1:].reshape(-1, 28,28)
        # x_after = np.array(self.after_data.iloc[index])[1:].reshape(-1, 28,28)
        x_before = self.before_data[index]
        x_after = self.after_data[index]
        x_before = x_before / 255
        x_after = x_after / 255
        x_before = torch.FloatTensor(x_before).to(dtype=torch_dtype)
        x_after = torch.FloatTensor(x_after).to(dtype=torch_dtype)
        return (x_before, x_after)

    def __len__(self):
        len_before = len(self.before_data)
        len_after = len(self.after_data)
        assert(len_before == len_after)
        return len_before

if args.dataset == "mnist":
  dataset_train = MNISTDataset(args)


# Create Dataloader from dataset
data_train = DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.cuda), pin_memory=args.cuda
)

class Expert(nn.Module):

    def __init__(self, args):
        super(Expert, self).__init__()
        self.args = args

        # Architecture
        def blockConv(k, in_feat, out_feat, BN = True, sigmoid = False):
          layers = [nn.Conv2d(in_feat, out_feat, k, padding='same')]
          if BN:
            layers.append(nn.BatchNorm2d(out_feat))
          if sigmoid:
            layers.append(nn.Sigmoid())
          else:
            layers.append(nn.ELU(0.2))
          return layers

        if self.args.dataset == 'mnist':
            self.model = nn.Sequential(
                *blockConv(3, 1, 32),
                *blockConv(3, 32, 32),
                *blockConv(3, 32, 32),
                *blockConv(3, 32, 32),
                *blockConv(3, 32, 1, BN = False, sigmoid=True),
            )
        else:
            raise NotImplementedError

    def forward(self, input):
        output = self.model(input)
        return output


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args

        def blockConv(k, in_feat, out_feat, sigmoid = False):
          layers = [nn.Conv2d(in_feat, out_feat, k, padding='same')]
          if sigmoid:
            layers.append(nn.Sigmoid())
          else:
            layers.append(nn.ELU(0.2))
          return layers

        # Architecture
        if self.args.dataset == 'mnist':
            self.model = nn.Sequential(
                *blockConv(3, 1, 16),
                *blockConv(3, 16, 16),
                *blockConv(3, 16, 32),
                nn.AvgPool2d(2, 2),
                *blockConv(3, 32, 32),
                *blockConv(3, 32, 64),
                nn.AvgPool2d(2, 2),
                *blockConv(3, 64, 64),
                *blockConv(3, 64, 64),
                nn.AvgPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(576, 25),
                nn.ELU(0.2),
                nn.Linear(25, 1),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError

    def forward(self, input):
        validity = self.model(input)
        return validity

def init_weights(model, path):
    pre_trained_dict = torch.load(path, map_location=lambda storage, loc: storage)
    for layer in pre_trained_dict.keys():
        model.state_dict()[layer].copy_(pre_trained_dict[layer])
    for param in model.parameters():
        param.requires_grad = True

def initialize_expert(epochs, expert, i, optimizer, loss, data_train, args, writer):
    print("Initializing expert [{}] as identity on preturbed data".format(i+1))
    expert.train()

    for epoch in range(epochs):
        total_loss = 0
        n_samples = 0
        for batch in data_train:
            x_canonical, x_transf = batch
            batch_size = x_canonical.size(0)
            n_samples += batch_size
            x_transf = x_transf.to(args.device).unsqueeze(1)
            x_canonical = x_canonical.to(args.device).unsqueeze(1)
            x_hat = expert(x_transf)
            loss_rec = loss(x_hat, x_transf)
            total_loss += loss_rec.item()*batch_size
            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()

        # Loss
        mean_loss = total_loss/n_samples
        print("initialization epoch [{}] expert [{}] loss {:.4f}".format(
            epoch+1, i+1, mean_loss))
        writer.add_scalar('expert_{}_initialization_loss'.format(
            i+1), mean_loss, epoch+1)
        if mean_loss < 0.002:
            break

    torch.save(expert.state_dict(), checkpt_dir +
               '/{}_E_{}_init.pth'.format(args.name, i + 1))
    
def train_system(epoch, experts, discriminator, optimizers_E, optimizer_D, criterion, data_train, args, writer):
    print("checkpoint 0")
    discriminator.train()
    print("checkpoint 1")
    for i, expert in enumerate(experts):
        expert.train()

    print("checkpoint 2")

    # Labels for canonical vs transformed samples
    canonical_label = 1
    transformed_label = 0

    # Keep track of losses
    total_loss_D_canon = 0
    total_loss_D_transformed = 0
    n_samples = 0
    total_loss_expert = [0 for i in range(len(experts))]
    total_samples_expert = [0 for i in range(len(experts))]
    expert_scores_D = [0 for i in range(len(experts))]
    expert_winning_samples_idx = [[] for i in range(len(experts))]

    dataset = [data for data in data_train]
    for itr in range(args.iterations):
      randIDX = np.random.randint(0, len(dataset), 1)[0]
      x_canon, x_transf = dataset[randIDX]
      x_canon = x_canon.unsqueeze(1)
      x_transf = x_transf.unsqueeze(1)
      n_samples = x_canon.shape[0]


      canon_out = discriminator(x_canon)
      loss_target = torch.ones(canon_out.shape)
      loss_D = criterion(canon_out, loss_target)
      total_loss_D_canon = loss_D.item() * n_samples
      optimizer_D.zero_grad()
      loss_D.backward()
      
      scores = []
      loss_D_transformed = 0
      loss_target = loss_target.fill_(0)
      for expert in experts:
        score = discriminator(expert(x_transf))
        loss_D_transformed += criterion(score, loss_target)
        scores.append(score)
      loss_D_transformed = loss_D_transformed / args.num_experts
      total_loss_D_transformed = loss_D_transformed.item() * n_samples
      loss_D_transformed.backward()
      optimizer_D.step()

      scores = torch.cat(scores, dim = 1)
      scores_argmax = torch.argmax(scores, axis=1, keepdims=True)
      for exp_i in range(len(experts)):
        wins = torch.nonzero((scores_argmax[:] == exp_i).to(torch.long).squeeze())
        if wins.shape[0] > 0:
          total_samples_expert[exp_i] += wins.shape[0]
          expi_scores = discriminator(experts[exp_i](x_transf[wins].squeeze(1)))
          loss_target = torch.ones(expi_scores.shape)
          loss_expi = criterion(expi_scores, loss_target)
          total_loss_expert[exp_i] += loss_expi.item() * wins.shape[0]
          optimizers_E[exp_i].zero_grad()
          loss_expi.backward(retain_graph = True)
          optimizers_E[exp_i].step()
          expert_scores_D[exp_i] += expi_scores.squeeze().sum().item()
      
      # Logging
      epoch = itr
      mean_loss_D_generated = total_loss_D_transformed / n_samples
      mean_loss_D_canon = total_loss_D_canon / n_samples
      print("epoch [{}] loss_D_transformed {:.4f}".format(
        epoch + 1, mean_loss_D_generated))
      print("epoch [{}] loss_D_canon {:.4f}".format(
        epoch + 1, mean_loss_D_canon))
      writer.add_scalar('loss_D_canonical', mean_loss_D_canon, epoch + 1)
      writer.add_scalar('loss_D_transformed', mean_loss_D_generated, epoch + 1)
      for i in range(len(experts)):
        print("epoch [{}] expert [{}] n_samples {}".format(
          epoch + 1, i + 1, total_samples_expert[i]))
        writer.add_scalar('expert_{}_n_samples'.format(
          i + 1), total_samples_expert[i], epoch + 1)
        writer.add_text('expert_{}_winning_samples'.format(i + 1),
                        ":".join([str(j) for j in expert_winning_samples_idx[i]]), epoch + 1)
        if total_samples_expert[i] > 0:
          mean_loss_expert = total_loss_expert[i] / total_samples_expert[i]
          mean_expert_scores = expert_scores_D[i] / total_samples_expert[i]
          print("epoch [{}] expert [{}] loss {:.4f}".format(
            epoch + 1, i + 1, mean_loss_expert))
          print("epoch [{}] expert [{}] scores {:.4f}".format(
            epoch + 1, i + 1, mean_expert_scores))
          writer.add_scalar('expert_{}_loss'.format(
            i + 1), mean_loss_expert, epoch + 1)
          writer.add_scalar('expert_{}_scores'.format(
            i + 1), mean_expert_scores, epoch + 1)

# Models Experts
experts = [Expert(args).to(args.device) for i in range(args.num_experts)]
# Losses
loss_initial = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.BCELoss(reduction='mean')

# Initialize Experts as approximately Identity on Transformed Data
for i, expert in enumerate(experts):
    if args.load_initialized_experts:
        path = os.path.join(checkpt_dir,
                            args.model_for_initialized_experts + '_E_{}_init.pth'.format(i+1))
        init_weights(expert, path)
    else:
        if args.optimizer_initialize == 'adam':
            optimizer_E = torch.optim.Adam(expert.parameters(), lr=args.learning_rate_initialize,
                                            weight_decay=args.weight_decay)
        else:
            raise NotImplementedError
        initialize_expert(args.epochs_init, expert, i,
                          optimizer_E, loss_initial, data_train, args, writer)

for i,expert in enumerate(experts):
  if args.inference_experts:
    path = os.path.join(checkpt_dir,
                        args.model_for_trained_experts + '_E_{}.pth'.format(i + 1))
    init_weights(expert, path)

# Model Desc
discriminator = Discriminator(args).to(args.device)
optimizers_E = []
for i in range(args.num_experts):
    if args.optimizer_experts == 'adam':
        optimizer_E = torch.optim.Adam(experts[i].parameters(), lr=args.learning_rate_expert,
                                        weight_decay=args.weight_decay)
    elif args.optimizer_experts == 'sgd':
        optimizer_E = torch.optim.SGD(experts[i].parameters(), lr=args.learning_rate_expert,
                                      weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    optimizers_E.append(optimizer_E)
if args.optimizer_discriminator == 'adam':
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate_discriminator,
                                    weight_decay=args.weight_decay)
elif args.optimizer_discriminator == 'sgd':
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=args.learning_rate_discriminator,
                                  weight_decay=args.weight_decay)


if not args.inference_experts:
  epoch = 1
  train_system(epoch, experts, discriminator, optimizers_E,
               optimizer_D, criterion, data_train, args, writer)
  torch.save(discriminator.state_dict(), checkpt_dir +
             '/{}_D.pth'.format(args.name))
  for i in range(args.num_experts):
    torch.save(experts[i].state_dict(), checkpt_dir +
               '/{}_E_{}.pth'.format(args.name, i+1))


import matplotlib.pyplot as plt
from PIL import Image

data_set = data_train.dataset
n_samples = len(data_set)

sample_count = 5
outs = []
for i in range(sample_count):
  random_index = int(np.random.random()*n_samples)
  single_example = data_set[random_index]
  x_canonical, x_transf = single_example

  x_transf_cuda = x_transf.to(args.device).unsqueeze(0).unsqueeze(0)
  out = [expert(x_transf_cuda).cpu().detach().squeeze().numpy()
          for expert in experts]
  pwidth = 3
  cval = 50
  out = [x_canonical.numpy(), x_transf.numpy()] + out
  out = [np.pad(item, pad_width=pwidth, constant_values=cval)
          for item in out
  ]
  # pcon = np.pad(x_canonical.numpy(), pad_width=pwidth)
  # pxf = np.pad(x_transf.numpy(), pad_width=pwidth)
  # out = [pcon, pxf] + out
  out_s = np.concatenate(out, axis = 1)
  outs.append(out_s)

sample = np.concatenate(outs, axis = 0)
sample = sample * 255
img = Image.fromarray(sample)
img.show()
