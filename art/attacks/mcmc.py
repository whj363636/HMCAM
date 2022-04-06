# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the Projected Gradient Descent attack `MCMC` as an iterative method in which,
after each iteration, the perturbation is projected on an lp-ball of specified radius (in addition to clipping the
values of the adversarial sample so that it lies in the permitted data range). This is the attack proposed by Madry et
al. for adversarial training.

| Paper link: https://arxiv.org/abs/1706.06083
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import torch
from art import NUMPY_DTYPE
from art.classifiers.classifier import ClassifierGradients
from art.attacks.fast_gradient import FastGradientMethod
from art.utils import compute_success, get_labels_np_array, random_sphere, projection, check_and_transform_label_format

logger = logging.getLogger(__name__)


class MCMC(FastGradientMethod):
    """
    The Projected Gradient Descent attack is an iterative method in which,
    after each iteration, the perturbation is projected on an lp-ball of specified radius (in
    addition to clipping the values of the adversarial sample so that it lies in the permitted
    data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """
    attack_params = FastGradientMethod.attack_params + ['max_iter']

    def __init__(self, classifier, norm=np.inf, eps=.3, eps_step=0.03, max_iter=100, targeted=False, num_random_init=0, decay_factor=1.0,
                 batch_size=1, writer=None):
        """
        Create a :class:`.MCMC` instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :type targeted: `bool`
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
            starting at the original input.
        :type num_random_init: `int`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super(MCMC, self).__init__(classifier, norm=norm, eps=eps, eps_step=eps_step,
                                                       targeted=targeted, num_random_init=num_random_init,
                                                       batch_size=batch_size, minimal=False)
        if not isinstance(classifier, ClassifierGradients):
            raise (TypeError('For `' + self.__class__.__name__ + '` classifier must be an instance of '
                             '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                             + str(classifier.__class__.__bases__) + '.'))

        kwargs = {'max_iter': max_iter}
        MCMC.set_params(self, **kwargs)

        self._project = True

        self.decay_factor = decay_factor
        self.writer = writer

        self.beta1, self.beta2 = 0.95, 0.999
        self.i = 0
        self.count = 5

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.classifier.nb_classes())

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError('Target labels `y` need to be provided for a targeted attack.')

            # Use model predictions as correct outputs
            targets = get_labels_np_array(self.classifier.predict(x, batch_size=self.batch_size))
        else:
            targets = y

        adv_x = x_init = x.astype(NUMPY_DTYPE)
        sample = np.zeros_like(adv_x)
        samples_seq = []
        v0 = v = random_sphere(x.shape[0], np.prod(x.shape[1:]), self.eps, self.norm).reshape(x.shape).astype(NUMPY_DTYPE)

        for j in range(max(1, self.num_random_init)):
            adv_x = x.astype(NUMPY_DTYPE)
            exp_avg = np.zeros_like(adv_x)
            exp_avg_sq = np.zeros_like(adv_x)
            max_exp_avg_sq = np.zeros_like(adv_x)
            self.i = 0

            for i_max_iter in range(self.max_iter):
                self.i = self.i + 1
                adv_x, v, exp_avg, exp_avg_sq, max_exp_avg_sq = self._compute(adv_x, x_init, targets, self.eps, self.eps_step, self._project, self.num_random_init > 0 and i_max_iter == 0, v, exp_avg, exp_avg_sq, max_exp_avg_sq)
                rate = 100 * compute_success(self.classifier, x_init, targets, adv_x, self.targeted, batch_size=self.batch_size)
                if not self.writer is None: 
                    self.writer.add_scalar('Success Attack Rate', scalar_value=rate, global_step=i_max_iter)

            samples = self._MH_correction(sample, x, adv_x, targets, v0, v)
            samples_seq.append(samples)
            x, v0 = adv_x, v

        return samples_seq

    def _MH_correction(self, sample, x, adv_x, targets, v0, v):
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x[batch_index_1:batch_index_2]
            adv_batch = adv_x[batch_index_1:batch_index_2]
            batch_labels = targets[batch_index_1:batch_index_2]

            orig = self.classifier.loss_generate(batch, batch_labels) + np.abs(v0[batch_index_1:batch_index_2])
            current = self.classifier.loss_generate(adv_batch, batch_labels) + np.abs(v[batch_index_1:batch_index_2])
            p_accept = np.minimum(1.0, np.exp(current - orig))
            
            seed = np.random.uniform(size=p_accept.shape).reshape(p_accept.shape).astype(NUMPY_DTYPE)
            mask = (p_accept > seed).astype(np.float32)
            sample[batch_index_1:batch_index_2] = adv_batch * mask + batch * (1-mask)
        
        return sample

    def _compute_perturbation(self, batch, batch_labels, random_init, v, exp_avg, exp_avg_sq, max_exp_avg_sq):
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        bias_correction1 = 1 - self.beta1 ** (self.i)
        bias_correction2 = 1 - self.beta2 ** (self.i)
        step_size = self.eps_step / bias_correction1

        # if random_init:
            # grad = self.classifier.loss_gradient(batch, batch_labels) * (1 - 2 * int(self.targeted))
            # v += 0.5 * step_size * grad
            # batch = batch + self._apply_norm(batch, v)

        grad = self.classifier.loss_gradient(batch, batch_labels) * (1 - 2 * int(self.targeted))
        exp_avg = exp_avg * self.beta1 + (1 - self.beta1) * grad
        exp_avg_sq = exp_avg_sq * self.beta2 + (1 - self.beta2) * grad * grad
        max_exp_avg_sq = np.maximum(max_exp_avg_sq, exp_avg_sq)
        denom = (np.sqrt(max_exp_avg_sq) / np.sqrt(bias_correction2)) + tol
        
        v = exp_avg

        perturbation = self._apply_norm(batch, v) * np.minimum(self.eps, step_size / denom)

        return perturbation, exp_avg, exp_avg_sq, max_exp_avg_sq, v

    def _apply_norm(self, batch, v):
        tol = 10e-8
        if self.norm == np.inf:
            perturbation = np.sign(v)
        elif self.norm == 1:
            ind = tuple(range(1, len(batch.shape)))
            perturbation = v / (np.sum(np.abs(v), axis=ind, keepdims=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(batch.shape)))
            perturbation = v / (np.sqrt(np.sum(np.square(v), axis=ind, keepdims=True)) + tol)
        assert batch.shape == v.shape

        return perturbation

    def _apply_perturbation(self, batch, perturbation):
        batch += perturbation

        if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
            batch = np.clip(batch, clip_min, clip_max)

        return batch

    def _compute(self, x, x_init, y, eps, eps_step, project, random_init, v, exp_avg, exp_avg_sq, max_exp_avg_sq):
        x_adv = x.astype(NUMPY_DTYPE)

        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation, batch_exp_avg, batch_exp_avg_sq, batch_max_exp_avg_sq, batch_v = self._compute_perturbation(batch, batch_labels, random_init, v[batch_index_1:batch_index_2], exp_avg[batch_index_1:batch_index_2], exp_avg_sq[batch_index_1:batch_index_2], max_exp_avg_sq[batch_index_1:batch_index_2])

            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation)
            # batch_v = batch_v + 0.5 * self.eps_step * self.classifier.loss_gradient(x_adv[batch_index_1:batch_index_2], batch_labels) * (1 - 2 * int(self.targeted))

            exp_avg[batch_index_1:batch_index_2] = batch_exp_avg
            exp_avg_sq[batch_index_1:batch_index_2] = batch_exp_avg_sq
            max_exp_avg_sq[batch_index_1:batch_index_2] = batch_max_exp_avg_sq
            v[batch_index_1:batch_index_2] = batch_v

            if project:
                perturbation = projection(x_adv[batch_index_1:batch_index_2] - x_init[batch_index_1:batch_index_2], eps, self.norm)
                x_adv[batch_index_1:batch_index_2] = x_init[batch_index_1:batch_index_2] + perturbation

        return x_adv, v, exp_avg, exp_avg_sq, max_exp_avg_sq


    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param norm: Order of the norm. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
            starting at the original input.
        :type num_random_init: `int`
        :param batch_size: Batch size
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(MCMC, self).set_params(**kwargs)

        if self.eps_step > self.eps:
            raise ValueError('The iteration step `eps_step` has to be smaller than the total attack `eps`.')

        if self.max_iter <= 0:
            raise ValueError('The number of iterations `max_iter` has to be a positive integer.')

        return True