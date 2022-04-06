from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random
import numpy as np
import torch
import torch.nn as nn
from art.attacks.attack import Attack
from art.utils import get_labels_np_array, random_sphere

logger = logging.getLogger(__name__)


def input_diversity(inputs, prob=0.5):
    size = inputs.shape[1]
    if np.random.uniform(0, 1, size=1) < prob:
        return inputs
    else:
        rnd = int(np.random.uniform(size, size*2, size=1)[0])
        h_rem = size*2 - rnd
        w_rem = size*2 - rnd
        pad_top = int(np.random.uniform(0, h_rem, size=1)[0])
        pad_left = int(np.random.uniform(0, w_rem, size=1)[0])
        pad_bottom = h_rem - pad_top
        pad_right = w_rem - pad_left

        rescaled = nn.Upsample(size=[rnd, rnd], mode='nearest')(torch.from_numpy(inputs))
        padded = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))(rescaled)

        return padded.numpy()

class DiversityMomentumIterative(Attack):
    attack_params = Attack.attack_params + ['norm', 'eps', 'targeted', 'random_init', 'batch_size', 'eps_step', 'max_iter', 'decay_factor']

    def __init__(self, classifier, norm=np.inf, eps=.3, eps_step=0.06, max_iter=20, targeted=False, random_init=False, decay_factor=1.0,
                 batch_size=128, expectation=None):
        """
        The Momentum Iterative Method (Dong et al. 2017). This method won
        the first places in NIPS 2017 Non-targeted Adversarial Attacks and
        Targeted Adversarial Attacks. The original paper used hard labels
        for this attack; no label smoothing.
        Paper link: https://arxiv.org/pdf/1710.06081.pdf
        :param model: cleverhans.model.Model
        :param sess: optional tf.Session
        :param dtypestr: dtype of the data
        :param kwargs: passed through to super constructor
        """
        super(DiversityMomentumIterative, self).__init__(classifier, expectation=expectation)

        self.norm = norm
        self.eps = eps
        self.targeted = targeted
        self.random_init = random_init
        self.batch_size = batch_size

        if eps_step > eps:
            raise ValueError('The iteration step `eps_step` has to be smaller than the total attack `eps`.')
        self.eps_step = eps_step

        if max_iter <= 0:
            raise ValueError('The number of iterations `max_iter` has to be a positive integer.')
        self.max_iter = int(max_iter)

        self.decay_factor = decay_factor
        self._project = True

    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param norm: Order of the norm. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param y: The labels for the data `x`. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
                  Labels should be one-hot-encoded.
        :type y: `np.ndarray`
        :param random_init: Whether to start at the original input or a random point within the epsilon ball
        :type random_init: `bool`
        :param batch_size: Batch size
        :type batch_size: `int`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        from art.utils import projection

        self.set_params(**kwargs)

        if 'y' not in kwargs or kwargs[str('y')] is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError('Target labels `y` need to be provided for a targeted attack.')

            # Use model predictions as correct outputs
            targets = get_labels_np_array(self._predict(x))
        else:
            targets = kwargs['y']
        target_labels = np.argmax(targets, axis=1)

        if self.random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:])
            adv_x = x.copy() + random_sphere(n, m, self.eps, self.norm).reshape(x.shape)
        else:
            adv_x = x.copy()

        momentum = np.zeros_like(adv_x)

        for i in range(self.max_iter):
            adv_x, momentum = self._compute(adv_x, targets, self.eps_step, momentum)

            if self._project:
                noise = projection(adv_x-x, self.eps, self.norm)
                adv_x = x + noise

            adv_preds = np.argmax(self._predict(adv_x), axis=1)
            if self.targeted:
                rate = np.sum(adv_preds == target_labels) / adv_x.shape[0]
            else:
                rate = np.sum(adv_preds != target_labels) / adv_x.shape[0]

            print('At iteration {:.0f}, test accuracy of DiversityMomentumIterative attack: {:.2f}%'.format(i, (1-rate)*100))

        return adv_x

    def _compute_perturbation(self, batch, batch_labels, momentum):
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self._loss_gradient(input_diversity(batch), batch_labels) * (1 - 2 * int(self.targeted))
        grad = grad / np.maximum(tol, np.mean(np.abs(grad), tuple(range(1, len(grad.shape))), keepdims=True))
        momentum = self.decay_factor * momentum + grad

        # Apply norm bound
        if self.norm == np.inf:
            perturbation = np.sign(momentum)
        elif self.norm == 1:
            ind = tuple(range(1, len(batch.shape)))
            perturbation = momentum / (np.sum(np.abs(momentum), axis=ind, keepdims=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(batch.shape)))
            perturbation = momentum / (np.sqrt(np.sum(np.square(momentum), axis=ind, keepdims=True)) + tol)
        assert batch.shape == perturbation.shape

        return perturbation, momentum

    def _apply_perturbation(self, batch, perturbation, eps):
        clip_min, clip_max = self.classifier.clip_values
        return np.clip(batch + eps * perturbation, clip_min, clip_max)

    def _compute(self, x, y, eps, momentum):
        adv_x = x.copy()
        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(adv_x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = adv_x[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation, batch_momentum = self._compute_perturbation(batch, batch_labels, momentum[batch_index_1:batch_index_2])

            # Apply perturbation and clip
            adv_x[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, eps)
            momentum[batch_index_1:batch_index_2] = batch_momentum
        return adv_x, momentum

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param norm: Order of the norm. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param random_init: Whether to start at the original input or a random point within the epsilon ball
        :type random_init: `bool`
        :param batch_size: Batch size
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(DiversityMomentumIterative, self).set_params(**kwargs)

        if self.eps_step > self.eps:
            raise ValueError('The iteration step `eps_step` has to be smaller than the total attack `eps`.')

        if self.max_iter <= 0:
            raise ValueError('The number of iterations `max_iter` has to be a positive integer.')

        return True