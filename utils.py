#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from torch import optim
import os


def make_optimizer(args, model, lr):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': args.momentum,
            'dampening': args.dampening,
            'nesterov': args.nesterov
        }
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon,
            'amsgrad': args.amsgrad
        }
    elif args.optimizer == 'ADAMAX':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'eps': args.epsilon,
            'momentum': args.momentum
        }
    else:
        raise Exception()

    kwargs['lr'] = lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)