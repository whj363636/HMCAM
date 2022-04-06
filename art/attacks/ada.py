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
This module implements the Projected Gradient Descent attack `Ada` as an iterative method in which,
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


class Ada(FastGradientMethod):
    """
    The Projected Gradient Descent attack is an iterative method in which,
    after each iteration, the perturbation is projected on an lp-ball of specified radius (in
    addition to clipping the values of the adversarial sample so that it lies in the permitted
    data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """
    attack_params = FastGradientMethod.attack_params + ['max_iter']

    def __init__(self, classifier, norm=np.inf, eps=.3, eps_step=0.1, max_iter=100, targeted=False, num_random_init=0, decay_factor=1.0,
                 batch_size=1, writer=None, name=None):
        """
        Create a :class:`.Ada` instance.

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
        super(Ada, self).__init__(classifier, norm=norm, eps=eps, eps_step=eps_step,
                                                       targeted=targeted, num_random_init=num_random_init,
                                                       batch_size=batch_size, minimal=False)
        if not isinstance(classifier, ClassifierGradients):
            raise (TypeError('For `' + self.__class__.__name__ + '` classifier must be an instance of '
                             '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                             + str(classifier.__class__.__bases__) + '.'))

        kwargs = {'max_iter': max_iter}
        Ada.set_params(self, **kwargs)

        self._project = True

        self.decay_factor = decay_factor
        self.writer = writer
        self.name = name

        self.beta1, self.beta2 = 0.95, 0.999
        self.radam_buffer = [[None,None,None] for ind in range(10)]
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

        adv_x = x.astype(NUMPY_DTYPE)
        exp_avg = np.zeros_like(adv_x)
        exp_avg_sq = np.zeros_like(adv_x)
        max_exp_avg_sq = np.zeros_like(adv_x)
        momentum = np.zeros_like(adv_x)
        slow_buffer = np.zeros_like(adv_x)

        for i_max_iter in range(self.max_iter):
            self.i = self.i + 1
            adv_x, momentum, exp_avg, exp_avg_sq, max_exp_avg_sq, slow_buffer = self._compute(adv_x, x, targets, self.eps, self.eps_step, self._project, self.num_random_init > 0 and i_max_iter == 0, momentum, exp_avg, exp_avg_sq, max_exp_avg_sq, slow_buffer)
            rate = 100 * compute_success(self.classifier, x, targets, adv_x, self.targeted, batch_size=self.batch_size)
            if not self.writer is None: 
                self.writer.add_scalar('Success Attack Rate', scalar_value=rate, global_step=i_max_iter)

            
        # logger.info('Success rate of attack: %.2f%%', rate_best if rate_best is not None else
        #             100 * compute_success(self.classifier, x, y, adv_x, self.targeted, batch_size=self.batch_size))

        return adv_x

    def _compute_perturbation(self, batch, batch_labels, momentum, exp_avg, exp_avg_sq, max_exp_avg_sq, slow_buffer):
        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        grad = self.classifier.loss_gradient(batch, batch_labels) * (1 - 2 * int(self.targeted))

        # Momentum 
        # grad = grad / np.maximum(tol, np.mean(np.abs(grad), tuple(range(1, len(grad.shape))), keepdims=True))
        # momentum = self.decay_factor * momentum + grad

        if self.name == 'RMSProp':
            bias_correction1 = 1 - self.beta1 ** (self.i)
            bias_correction2 = 1 - self.beta2 ** (self.i)

            exp_avg = exp_avg * self.beta1 + (1 - self.beta1) * grad
            exp_avg_sq = exp_avg_sq - (1 - self.beta2) * np.sign(exp_avg_sq-grad * grad) * grad * grad

            max_exp_avg_sq = np.maximum(max_exp_avg_sq, exp_avg_sq)
            denom = (np.sqrt(max_exp_avg_sq) / np.sqrt(bias_correction2)) + tol
            momentum = exp_avg / denom
            step_size = self.eps_step / bias_correction1

        elif self.name == 'AMSGrad':
            bias_correction1 = 1 - self.beta1 ** (self.i)
            bias_correction2 = 1 - self.beta2 ** (self.i)

            exp_avg = exp_avg * self.beta1 + (1 - self.beta1) * grad
            exp_avg_sq = exp_avg_sq * self.beta2 + (1 - self.beta2) * grad * grad

            max_exp_avg_sq = np.maximum(max_exp_avg_sq, exp_avg_sq)
            denom = (np.sqrt(max_exp_avg_sq) / np.sqrt(bias_correction2)) + tol
            momentum = exp_avg / denom
            step_size = self.eps_step / bias_correction1

        elif self.name == 'SAMSGrad':
            bias_correction1 = 1 - self.beta1 ** (self.i)
            bias_correction2 = 1 - self.beta2 ** (self.i)

            exp_avg = exp_avg * self.beta1 + (1 - self.beta1) * grad
            exp_avg_sq = exp_avg_sq * self.beta2 + (1 - self.beta2) * grad * grad
            
            max_exp_avg_sq = np.maximum(max_exp_avg_sq, exp_avg_sq)
            denom = np.sqrt(max_exp_avg_sq)
            momentum = exp_avg / torch.nn.Softplus(50)(torch.from_numpy(denom)).numpy()
            step_size = self.eps_step * np.sqrt(bias_correction2) / bias_correction1

        elif self.name == 'Ranger':
            exp_avg = exp_avg * self.beta1 + (1 - self.beta1) * grad
            exp_avg_sq = exp_avg_sq * self.beta2 + (1 - self.beta2) * grad * grad

            buffered = self.radam_buffer[int(self.i % 10)]
            if self.i == buffered[0]:
                N_sma, step_size = buffered[1], buffered[2]
            else:
                buffered[0] = self.i
                beta2_t = self.beta2 ** self.i
                N_sma_max = 2 / (1 - self.beta2) - 1
                N_sma = N_sma_max - 2 * self.i * beta2_t / (1 - beta2_t)
                buffered[1] = N_sma
                if N_sma > self.count:
                    step_size = self.eps_step * np.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - self.beta1 ** self.i)
                else:
                    step_size = self.eps_step / (1 - self.beta1 ** self.i)
                buffered[2] = step_size

            if N_sma > self.count:
                momentum = exp_avg / (np.sqrt(exp_avg_sq) + 1e-5)
            else:
                momentum = exp_avg

        # Apply norm bound
        if self.norm == np.inf:
            perturbation = np.sign(momentum)
        elif self.norm == 1:
            ind = tuple(range(1, len(batch.shape)))
            perturbation = momentum / (np.sum(np.abs(momentum), axis=ind, keepdims=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(batch.shape)))
            perturbation = momentum / (np.sqrt(np.sum(np.square(momentum), axis=ind, keepdims=True)) + tol)
        assert batch.shape == momentum.shape

        perturbation = perturbation * min(self.eps, step_size)

        return perturbation, momentum, exp_avg, exp_avg_sq, max_exp_avg_sq, slow_buffer

    def _apply_perturbation(self, batch, perturbation, eps_step):
        batch += perturbation

        if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
            batch = np.clip(batch, clip_min, clip_max)

        return batch

    def _compute(self, x, x_init, y, eps, eps_step, project, random_init, momentum, exp_avg, exp_avg_sq, max_exp_avg_sq, slow_buffer):
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:])
            x_adv = x.astype(NUMPY_DTYPE) + random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(NUMPY_DTYPE)

            if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
                clip_min, clip_max = self.classifier.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)
        else:
            x_adv = x.astype(NUMPY_DTYPE)

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]
            # self.i = self.i + 1
            # Get perturbation
            perturbation, batch_momentum, batch_exp_avg, batch_exp_avg_sq, batch_max_exp_avg_sq, batch_slow_buffer = self._compute_perturbation(batch, batch_labels, momentum[batch_index_1:batch_index_2], exp_avg[batch_index_1:batch_index_2], exp_avg_sq[batch_index_1:batch_index_2], max_exp_avg_sq[batch_index_1:batch_index_2], slow_buffer[batch_index_1:batch_index_2])

            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, eps_step)
            momentum[batch_index_1:batch_index_2] = batch_momentum
            exp_avg[batch_index_1:batch_index_2] = batch_exp_avg
            exp_avg_sq[batch_index_1:batch_index_2] = batch_exp_avg_sq
            max_exp_avg_sq[batch_index_1:batch_index_2] = batch_max_exp_avg_sq
            slow_buffer[batch_index_1:batch_index_2] = batch_slow_buffer

            if project:
                perturbation = projection(x_adv[batch_index_1:batch_index_2] - x_init[batch_index_1:batch_index_2], eps, self.norm)
                x_adv[batch_index_1:batch_index_2] = x_init[batch_index_1:batch_index_2] + perturbation

        return x_adv, momentum, exp_avg, exp_avg_sq, max_exp_avg_sq, slow_buffer

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
        super(Ada, self).set_params(**kwargs)

        if self.eps_step > self.eps:
            raise ValueError('The iteration step `eps_step` has to be smaller than the total attack `eps`.')

        if self.max_iter <= 0:
            raise ValueError('The number of iterations `max_iter` has to be a positive integer.')

        return True