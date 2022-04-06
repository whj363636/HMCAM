from __future__ import division

import os, sys, pdb, shutil, time, random, copy
import os.path as osp
import glob
import re
import numpy as np
import argparse

# observe the distribution/heatmap of class before attack
def class_cluster_before(dir_path, cls):
  img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
  pattern = re.compile(r'([-\d]+)_(\d)_(\d)_(\d)') # imagenum_gt_pred_adv

  dataset = []
  for img_path in img_paths:
    num, gt, pred, adv_pred = map(int, pattern.search(img_path).groups())
    if pred == cls:
      dataset.append((img_path, cls))

  return dataset

# observe the distribution/heatmap of class after attack
def class_cluster_after(path, cls):
  img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
  pattern = re.compile(r'([-\d]+)_(\d)_(\d)_(\d)') # imagenum_gt_pred_adv

  dataset = []
  for img_path in img_paths:
    num, gt, pred, adv_pred = map(int, pattern.search(img_path).groups())
    if adv_pred == cls and not adv_pred == gt:
      dataset.append((img_path, cls))

  return dataset


def image_cluster(path):


class Market1501(object):
    dataset_dir = 'market1501'

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            pid_raw = pid
            if relabel: 
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid, pid_raw))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs



if __name__ == '__main__':
