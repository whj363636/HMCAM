# -*- coding: utf-8 -*- 
from __future__ import division
from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

import os, sys, pdb, shutil, time, random, copy
import os.path as osp
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time

from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.iterative_method import BasicIterativeMethod
from art.attacks.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.carlini import CarliniL2Method, CarliniLInfMethod
from art.attacks.momentum_iterative_method import MomentumIterative
from art.attacks.ada import Ada
from art.attacks.mcmc import MCMC
from art.classifiers import PyTorchClassifier, EnsembleClassifier
from art.defences import AdversarialTrainer
from art.utils import load_dataset
import models
import copy
import cv2

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# path
parser.add_argument('--data_path', type=str, help='Path to dataset')
parser.add_argument('--save_path', type=str, default='./snapshots/cifar10', help='Folder to save checkpoints and log.')
# 
parser.add_argument('--ak_type', default='', type=str, help='Type of attack')
parser.add_argument('--defense', default='', type=str, help='defense')
parser.add_argument('--eps', type=float, default=0.3, help='l_inf magnitude of perturbation')
parser.add_argument('--itr', type=int, default=100, help='iteration of attack')
parser.add_argument('--num_sample', type=int, default=1, help='num_sample of MCMC')
parser.add_argument('--stepsize', type=float, default=0.3, help='step size')

parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
parser.add_argument('--arches', nargs='+', default=['resnet32'])
parser.add_argument('--pretrained', nargs='+', default=[''], type=str, help='path to latest checkpoint (default: none)')

# Optimization options
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
args.arches = ['resnet32']
args.pretrained = ['./snapshots/cifar10/resnet32/model_best.pth.tar']

if args.manualSeed is None:
  args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
  torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

if args.dataset == 'cifar10':
  mean = [x / 255 for x in [125.3, 123.0, 113.9]]
  std = [x / 255 for x in [63.0, 62.1, 66.7]]
  num_classes = 10
  input_shape = (32, 32)
elif args.dataset == 'cifar100':
  mean = [x / 255 for x in [129.3, 124.1, 112.4]]
  std = [x / 255 for x in [68.2, 65.4, 70.4]]
  num_classes = 100
  input_shape = (32, 32)
elif args.dataset == 'mnist':
  mean = [0.1307,]
  std = [0.3081,]
  num_classes = 10
  input_shape = (28, 28)
else:
  assert False, "Unknow dataset : {}".format(args.dataset)

def main():
  # Init logger
  if not osp.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(osp.join(args.save_path, 'log.txt'), 'w')
  state = {k: v for k, v in args._get_kwargs()}

  # Init dataset
  if not osp.isdir(args.data_path): os.makedirs(args.data_path)
  (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str(args.dataset), mean, std)

  criterion = torch.nn.CrossEntropyLoss()

  if args.dataset in ['cifar10', 'cifar100']: 
    net = models.__dict__[args.arches[0]](num_classes=num_classes, dataset=args.dataset)
  else:
    net = models.mnistnet(num_classes=num_classes)

  if args.use_cuda:
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
  optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)
  classifier = PyTorchClassifier(clip_values=(min_, max_), model=net, loss=criterion, optimizer=optimizer, input_shape=input_shape, nb_classes=num_classes, channel_index=1, defences=None, preprocessing=(0, 1))
  
  if osp.isfile(args.pretrained[0]):
    checkpoint = torch.load(args.pretrained[0], pickle_module=pickle)
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
  
  if args.ak_type == 'FGSM':
    adv_crafter = FastGradientMethod(classifier, eps=args.eps, targeted=False, batch_size=args.batch_size)
  elif args.ak_type == 'BIM':
    adv_crafter = BasicIterativeMethod(classifier, eps=args.eps, max_iter=args.itr, targeted=False, batch_size=args.batch_size, writer=None)
  elif args.ak_type == 'PGD':  
    adv_crafter = ProjectedGradientDescent(classifier, eps=args.eps, max_iter=args.itr, targeted=False, batch_size=args.batch_size, num_random_init=20)
  # elif args.ak_type == 'Momentum':  
  #   adv_crafter = MomentumIterative(classifier, eps=args.eps, eps_step=args.stepsize, max_iter=args.itr, targeted=False, batch_size=args.batch_size, writer=None)
  elif args.ak_type == 'MCMC':  
    adv_crafter = MCMC(classifier, eps=args.eps, eps_step=args.stepsize, max_iter=args.itr, targeted=False, num_random_init=args.num_sample, batch_size=args.batch_size, writer=None)

  if args.ak_type == 'MCMC':
    n,h,w,c = x_train.shape
    x_train_adv = np.reshape(adv_crafter.generate(x_train[:50]), (50*args.num_sample,h,w,c))
    x_train = np.append(x_train, x_train_adv, axis=0)
    y_train_adv = np.repeat(y_train[:50], args.num_sample, axis=0)
    y_train = np.append(y_train, y_train_adv, axis=0)
  else:
    x_train_adv = adv_crafter.generate(x_train[:50*args.num_sample])
    x_train = np.append(x_train, x_train_adv, axis=0)
    y_train = np.append(y_train, y_train[:50*args.num_sample], axis=0)

  classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)
  # trainer = AdversarialTrainer(classifier, adv_crafter, ratio=0.8)
  # trainer.fit(x_train, y_train, nb_epochs=args.epochs, batch_size=args.batch_size)

  # attack
  adv_crafter = MomentumIterative(classifier, eps=args.eps, eps_step=args.stepsize, max_iter=20, targeted=False, batch_size=args.batch_size, writer=None)
  x_test_adv = adv_crafter.generate(x=x_test, random_init=True)

  preds = np.argmax(classifier.predict(x_test), axis=1)
  acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]

  adv_preds = np.argmax(classifier.predict(x_test_adv), axis=1)
  acc = np.sum(adv_preds != preds) / y_test.shape[0]
  print_log("=> Success Attack Rate: {:.2f}%.\n".format(acc*100), log)

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()

if __name__ == '__main__':
  main()