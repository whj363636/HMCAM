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
from torch.utils.tensorboard import SummaryWriter
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

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# path
parser.add_argument('--data_path', type=str, help='Path to dataset')
parser.add_argument('--vis_path', type=str, help='Path to save vis')
parser.add_argument('--save_path', type=str, default='./snapshots/cifar10', help='Folder to save checkpoints and log.')
# 
parser.add_argument('--ak_type', default='', type=str, help='Type of attack')
parser.add_argument('--defense', default='', type=str, help='defense')
parser.add_argument('--eps', type=float, default=0.3, help='l_inf magnitude of perturbation')
parser.add_argument('--itr', type=int, default=100, help='iteration of attack')
parser.add_argument('--num_sample', type=int, default=1, help='num_sample of MCMC')
parser.add_argument('--stepsize', type=float, default=0.3, help='step size')
parser.add_argument('--usevis', action='store_true', default=False, help='whether to save vis')

parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
parser.add_argument('--arches', nargs='+', default=['resnet32'])
parser.add_argument('--pretrained', nargs='+', default=[''], type=str, help='path to latest checkpoint (default: none)')

# Optimization options
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
args.arches = ['resnet32', 'vgg16', 'googlenet', 'resnext29_8_64', 'densenet_cifar']#, 'resnet32', 'resnext29_8_64']
args.pretrained = ['./snapshots/cifar10/resnet32/model_best.pth.tar', './snapshots/cifar10/vgg16/model_best.pth.tar', 
                  './snapshots/cifar10/googlenet/model_best.pth.tar', './snapshots/cifar10/resnext29_8_64/model_best.pth.tar', 
                  './snapshots/cifar10/densenet_cifar/model_best.pth.tar']
                  # './snapshots/cifar10/resnet32_adv/model_best.pth.tar',
                  # './snapshots/cifar10/resnext29_8_64_adv/model_best.pth.tar']

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
  classifiers = [None] * 14

  # Init logger
  if not osp.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(osp.join(args.save_path, 'log.txt'), 'w')
  state = {k: v for k, v in args._get_kwargs()}

  # Init dataset
  if not osp.isdir(args.data_path): os.makedirs(args.data_path)
  (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str(args.dataset), mean, std)

  criterion = torch.nn.CrossEntropyLoss()

  for i in range(len(args.arches)):
    if args.dataset in ['cifar10', 'cifar100']: 
      net = models.__dict__[args.arches[i]](num_classes=num_classes, dataset=args.dataset)
    else:
      net = models.mnistnet(num_classes=num_classes)

    if args.use_cuda:
      net = net.cuda()
      net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)
    classifier = PyTorchClassifier(clip_values=(min_, max_), model=net, loss=criterion, optimizer=optimizer, input_shape=input_shape, nb_classes=num_classes, channel_index=1, defences=None, preprocessing=(0, 1))
    
    if osp.isfile(args.pretrained[i]):
      checkpoint = torch.load(args.pretrained[i], pickle_module=pickle)
      net.load_state_dict(checkpoint['state_dict'], strict=False)
      optimizer.load_state_dict(checkpoint['optimizer'])
    
    classifiers[i] = classifier

  classifiers[5] = EnsembleClassifier(classifiers=[classifiers[i] for i in (1,2,3,4)], clip_values=(min_, max_))
  classifiers[6] = EnsembleClassifier(classifiers=[classifiers[i] for i in (0,2,3,4)], clip_values=(min_, max_))
  classifiers[7] = EnsembleClassifier(classifiers=[classifiers[i] for i in (0,1,3,4)], clip_values=(min_, max_))
  classifiers[8] = EnsembleClassifier(classifiers=[classifiers[i] for i in (0,1,2,4)], clip_values=(min_, max_))
  classifiers[9] = EnsembleClassifier(classifiers=[classifiers[i] for i in (0,1,2,3)], clip_values=(min_, max_))

  # on adversarial examples
  # for i in (0,1,2,3,4):
  #   writer = None

  #   if args.ak_type == 'FGSM':
  #     adv_crafter = FastGradientMethod(classifiers[i], eps=args.eps, targeted=False, batch_size=args.batch_size)
  #   elif args.ak_type == 'BIM':
  #     adv_crafter = BasicIterativeMethod(classifiers[i], eps=args.eps, max_iter=args.itr, targeted=False, batch_size=args.batch_size, writer=writer)
  #   elif args.ak_type == 'PGD':  
  #     adv_crafter = ProjectedGradientDescent(classifiers[i], eps=args.eps, max_iter=args.itr, targeted=False, batch_size=args.batch_size, num_random_init=20, writer=writer)
  #   elif args.ak_type == 'Momentum':  
  #     adv_crafter = MomentumIterative(classifiers[i], eps=args.eps, max_iter=args.itr, targeted=False, batch_size=args.batch_size, writer=writer)
  #   elif args.ak_type == 'MCMC':  
  #     adv_crafter = MCMC(classifiers[i], eps=args.eps, eps_step=args.stepsize, max_iter=args.itr, targeted=False, num_random_init=args.num_sample, batch_size=args.batch_size, writer=writer)
  #   elif args.ak_type in ['RMSProp', 'AMSGrad', 'SAMSGrad', 'Ranger']:  
  #     adv_crafter = Ada(classifiers[i], eps=args.eps, eps_step=args.stepsize, max_iter=args.itr, targeted=False, batch_size=args.batch_size, writer=writer, name=args.ak_type)

  #   x_test_adv = adv_crafter.generate(x=x_test) # Maximum perturbation

  #   for j in (5,6):
  #     print_log("{} -> {}".format(args.arches[i], args.arches[j]), log)
  #     preds = np.argmax(classifiers[j].predict(x_test), axis=1)
  #     acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]

  #     print_log("=> Test accuracy: %.2f%%" % (acc * 100), log)
  #     adv_preds = np.argmax(classifiers[j].predict(x_test_adv), axis=1)
  #     acc = np.sum(adv_preds != preds) / y_test.shape[0]
  #     print_log("=> Success Attack Rate: {:.2f}%.\n".format(acc*100), log)


################### ensemble ########################
  ensemble_name = ['holdout_resnet32', 'holdout_vgg16', 'holdout_googlenet', 'holdout_resnext29_8_64', 'holdout_densenet']
  for i in range(5,10):
    writer = None

    if args.ak_type == 'FGSM':
      adv_crafter = FastGradientMethod(classifiers[i], eps=args.eps, targeted=False, batch_size=args.batch_size)
    elif args.ak_type == 'BIM':
      adv_crafter = BasicIterativeMethod(classifiers[i], eps=args.eps, max_iter=args.itr, targeted=False, batch_size=args.batch_size, writer=writer)
    elif args.ak_type == 'PGD':  
      adv_crafter = ProjectedGradientDescent(classifiers[i], eps=args.eps, max_iter=args.itr, targeted=False, batch_size=args.batch_size, num_random_init=20, writer=writer)
    elif args.ak_type == 'Momentum':  
      adv_crafter = MomentumIterative(classifiers[i], eps=args.eps, max_iter=args.itr, targeted=False, batch_size=args.batch_size, writer=writer)
    elif args.ak_type == 'MCMC':  
      adv_crafter = MCMC(classifiers[i], eps=args.eps, eps_step=args.stepsize, max_iter=args.itr, targeted=False, num_random_init=args.num_sample, batch_size=args.batch_size, writer=writer)
    elif args.ak_type in ['RMSProp', 'AMSGrad', 'SAMSGrad', 'Ranger']:  
      adv_crafter = Ada(classifiers[i], eps=args.eps, eps_step=args.stepsize, max_iter=args.itr, targeted=False, batch_size=args.batch_size, writer=writer, name=args.ak_type)

    x_test_adv = adv_crafter.generate(x=x_test) # Maximum perturbation

    for j in range(5,10):
      print_log("Ensemble: {} -> {}".format(ensemble_name[i-5], ensemble_name[j-5]), log)
      preds = np.argmax(classifiers[j].predict(x_test), axis=1)
      acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]

      print_log("=> Test accuracy: %.2f%%" % (acc * 100), log)
      adv_preds = np.argmax(classifiers[j].predict(x_test_adv), axis=1)
      acc = np.sum(adv_preds != preds) / y_test.shape[0]
      print_log("=> Success Attack Rate: {:.2f}%.\n".format(acc*100), log)

      print_log("Holdout: {} -> {}".format(ensemble_name[i-5], args.arches[j-5]), log)
      preds = np.argmax(classifiers[j-5].predict(x_test), axis=1)
      acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]

      print_log("=> Test accuracy: %.2f%%" % (acc * 100), log)
      adv_preds = np.argmax(classifiers[j-5].predict(x_test_adv), axis=1)
      acc = np.sum(adv_preds != preds) / y_test.shape[0]
      print_log("=> Success Attack Rate: {:.2f}%.\n".format(acc*100), log)


def visualize(x_test, gt, pred, adv_pred):
  print("=> Saving adversarial examples")
  path = gen_path(args.vis_path, [args.dataset + '_' + args.arch + '_' + args.ak_type])
  num, chan, h, w = x_test.shape
  for i in range(num):
    inputs = x_test[i:i+1, :, :, :]
    original = np.zeros_like(inputs)
    GT = str(np.argmax(gt[i:i+1, :], axis=1)[0])
    Before = str(pred[i:i+1][0])
    After = str(adv_pred[i:i+1][0])

    for c in range(chan): original[:, c, :, :] = (inputs[:, c, :, :] * std[c]) + mean[c]
    original = original * 255.0
    original = original.transpose((0,2,3,1)).squeeze(0)
    original = original.astype(np.uint8)
    cv2.imwrite(osp.join(path, '{}_{}_{}_{}.jpg'.format(i, GT, Before, After)), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))

def gen_path(path, suffix):
  for item in suffix:
    path = osp.join(path, item)
  if not osp.exists(path):
    os.makedirs(path)
  return path

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()

if __name__ == '__main__':
  main()