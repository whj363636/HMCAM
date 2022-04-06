#!/usr/bin/env python
#-*- coding: utf-8 -*-
#File:

import sys
import argparse
import tensorflow as tf
import tqdm
import numpy as np
import cv2
import os
import glob
import os.path as osp

from sklearn import metrics
from scipy import misc
from scipy.optimize import brentq
from scipy import interpolate

import lfw as lfw
import align.detect_face as FaceDet


class Model():

    def __init__(self, args):
        from models import inception_resnet_v1  # facenet model
        self.network = inception_resnet_v1
        self.num_iter = args.itr
        self.beta1, self.beta2 = 0.9, 0.999

        self.image_batch = tf.placeholder(tf.uint8, shape=[None, 160, 160, 3], name='images')

        image = (tf.to_float(self.image_batch) - 127.5) / 128.0
        prelogits, _ = self.network.inference(image, 1.0, False, bottleneck_layer_size=512)
        self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, 'models/20180402-114759/model-20180402-114759.ckpt-275')

    def compute_victim(self, lfw_160_path, name):
        imgfolder = os.path.join(lfw_160_path, name)
        assert os.path.isdir(imgfolder), imgfolder
        images = glob.glob(os.path.join(imgfolder, '*.png')) + glob.glob(os.path.join(imgfolder, '*.jpg'))
        image_batch = [cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1] for f in images]
        for img in image_batch:
            assert img.shape[0] == 160 and img.shape[1] == 160, \
                "--data should only contain 160x160 images. Please read the README carefully."
        embeddings = self.eval_embeddings(image_batch)
        self.victim_embeddings = embeddings
        return embeddings

    def structure(self, input_tensor):
        """
        Args:
            input_tensor: NHWC
        """
        rnd = tf.random_uniform((), 135, 160, dtype=tf.int32)
        rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        h_rem = 160 - rnd
        w_rem = 160 - rnd
        pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
        pad_right = w_rem - pad_left
        pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
        pad_bottom = h_rem - pad_top
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [
                        pad_left, pad_right], [0, 0]])
        padded.set_shape((input_tensor.shape[0], 160, 160, 3))
        output = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.9),
                         lambda: padded, lambda: input_tensor)
        return output

    def build_pgd_attack(self, eps):
        victim_embeddings = tf.constant(self.victim_embeddings, dtype=tf.float32)

        def cond(i, _, __, ___, ____, ______):
            """Iterate until number of iterations completed"""
            return tf.less(i, self.num_iter)
        
        def one_step_attack(i, adv, m, exp_avg, exp_avg_sq, max_exp_avg_sq):
            """
            core components of this attack are:
            (a) PGD adversarial attack (https://arxiv.org/pdf/1706.06083.pdf)
            (b) momentum (https://arxiv.org/pdf/1710.06081.pdf)
            (c) input diversity (https://arxiv.org/pdf/1803.06978.pdf)
            """
            orig_image = adv
            adv = self.structure(adv)
            adv = (adv - 127.5) / 128.0
            adv = adv + tf.random_uniform(tf.shape(adv), minval=-1e-2, maxval=1e-2)
            prelogits, _ = self.network.inference(adv, 1.0, False, bottleneck_layer_size=512)
            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

            embeddings = tf.reshape(embeddings[0], [512, 1])
            objective = tf.reduce_mean(tf.matmul(victim_embeddings, embeddings))  # to be maximized

            g, = tf.gradients(objective, orig_image)

            bias_correction1 = 1 - self.beta1 ** (i+1)
            bias_correction2 = 1 - self.beta2 ** (i+1)

            # Decay the first and second moment running average coefficient
            exp_avg = exp_avg * self.beta1 + (1 - self.beta1) * g
            exp_avg_sq = exp_avg_sq * self.beta2 + (1 - self.beta2) * g * g

            max_exp_avg_sq = tf.maximum(max_exp_avg_sq, exp_avg_sq)
            denom = tf.sqrt(max_exp_avg_sq) / tf.sqrt(bias_correction2)
            m = exp_avg / (denom + 1e-5)

            adv = tf.clip_by_value(orig_image + tf.sign(m) * tf.minimum(float(eps), 1 / bias_correction1), lower_bound, upper_bound)
            return i + 1, adv, m, exp_avg, exp_avg_sq, max_exp_avg_sq

        input = tf.to_float(self.image_batch)
        lower_bound = tf.clip_by_value(input - eps, 0, 255.)
        upper_bound = tf.clip_by_value(input + eps, 0, 255.)

        init_momentum = init_exp_avg = init_exp_avg_sq = init_max_exp_avg_sq = tf.zeros_like(input)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            _, adv, _, _, _, _ = tf.while_loop(cond, one_step_attack, (tf.zeros([]), input, init_momentum, init_exp_avg, init_exp_avg_sq, init_max_exp_avg_sq), back_prop=False, maximum_iterations=self.num_iter, parallel_iterations=1)
        
        self.adv_image = adv
        return adv

    def eval_attack(self, img):
        # img: single HWC image
        out = self.sess.run(
            self.adv_image, feed_dict={self.image_batch: [img]})[0]
        return out

    def eval_embeddings(self, batch_arr):
        return self.sess.run(self.embeddings, feed_dict={self.image_batch: batch_arr})

    def distance_to_victim(self, img):
        emb = self.eval_embeddings([img])
        dist = np.dot(emb, self.victim_embeddings.T).flatten()
        stats = np.percentile(dist, [10, 30, 50, 70, 90])
        return stats


class Detector():
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = FaceDet.create_mtcnn(sess, None)

    def detect(self, img):
        """
        img: rgb 3 channel
        """
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        bounding_boxes, _ = FaceDet.detect_face(
                img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
        num_face = bounding_boxes.shape[0]
        assert num_face == 1, num_face
        bbox = bounding_boxes[0][:4]  # xy,xy

        margin = 32
        x0 = np.maximum(bbox[0] - margin // 2, 0)
        y0 = np.maximum(bbox[1] - margin // 2, 0)
        x1 = np.minimum(bbox[2] + margin // 2, img.shape[1])
        y1 = np.minimum(bbox[3] + margin // 2, img.shape[0])
        x0, y0, x1, y1 = bbox = [int(k + 0.5) for k in [x0, y0, x1, y1]]
        cropped = img[y0:y1, x0:x1, :]
        scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
        return scaled, bbox


def validate_on_lfw(model, lfw_160_path):
    # Read the file containing the pairs used for testing
    pairs = lfw.read_pairs('validation-LFW-pairs.txt')
    # Get the paths for the corresponding images
    paths, actual_issame = lfw.get_paths(lfw_160_path, pairs)
    num_pairs = len(actual_issame)

    all_embeddings = np.zeros((num_pairs * 2, 512), dtype='float32')
    for k in tqdm.trange(num_pairs):
        img1 = cv2.imread(paths[k * 2], cv2.IMREAD_COLOR)[:, :, ::-1]
        img2 = cv2.imread(paths[k * 2 + 1], cv2.IMREAD_COLOR)[:, :, ::-1]
        batch = np.stack([img1, img2], axis=0)
        embeddings = model.eval_embeddings(batch)
        all_embeddings[k * 2: k * 2 + 2, :] = embeddings

    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(
        all_embeddings, actual_issame, distance_metric=1, subtract_mean=True)

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--data', help='path to MTCNN-aligned LFW dataset',
            default=os.path.expanduser('~/data/LFW/MTCNN_160'))
    parser.add_argument('--save_dir', type=str, default='adv_images', help='path to save model')

    parser.add_argument('--eps', type=int, default=16, help='maximum pixel perturbation')
    parser.add_argument('--itr', type=int, default=100, help='maximum pixel perturbation')

    parser.add_argument('--validate-lfw', action='store_true')
    parser.add_argument('--attack', help='input image to detect face and attack')
    parser.add_argument('--output', help='output image', default='output.png')

    parser.add_argument('--detect', help='input image to detect face')
    parser.add_argument('--attack-cropped', help='input 160x160 with aligned face to attack')
    parser.add_argument('--target', default='Arnold_Schwarzenegger')
    args = parser.parse_args()

    model = Model(args)

    if not osp.exists(args.save_dir): os.makedirs(args.save_dir)
    args.output = osp.join('adv_images', args.output)

    if args.validate_lfw:
        validate_on_lfw(model, args.data)
        sys.exit()
    if args.detect:
        det = Detector()
        img = cv2.imread(args.detect)[:, :, ::-1]
        scaled_face, bbox = det.detect(img)
        cv2.imwrite(args.output, scaled_face[:, :, ::-1])
        sys.exit()

    victim = model.compute_victim(args.data, args.target)
    print("Number of victim samples (the more the better): {}".format(len(victim)))
    model.build_pgd_attack(args.eps)
    if args.attack_cropped:
        img = cv2.imread(args.attack_cropped)[:, :, ::-1]
        out = model.eval_attack(img)
        cv2.imwrite(args.output, out[:, :, ::-1])

        print("Similarity of ORIG:", model.distance_to_victim(img))
        print("Similarity of ADV:", model.distance_to_victim(out[0]))
    elif args.attack:
        det = Detector()
        img = cv2.imread(args.attack)[:, :, ::-1]
        scaled_face, bbox = det.detect(img)

        print("ORIG detected box:", bbox)
        print("Similarity of ORIG:", model.distance_to_victim(scaled_face))

        attack_face = model.eval_attack(scaled_face)
        print("Similarity of ADV:", model.distance_to_victim(attack_face))
        attack_face_rescaled = misc.imresize(
                attack_face, (bbox[3] - bbox[1], bbox[2] - bbox[0]))
        img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = attack_face_rescaled
        cv2.imwrite(args.output, img[:, :, ::-1])

        # scaled_face, bbox = det.detect(img)
        # print("Re-detected box:", bbox)
        # print("Similarity of Re-detected ADV:", model.distance_to_victim(scaled_face))
