#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import os
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as lyr
import pickle as pkl
from fuel.schemes import ShuffledScheme

import utils


def load_mnist_dataset(dataset_path, args):

    with open(os.path.join(dataset_path, 'mnist.pkl'), 'rb') as f:
        train_set, valid_set, test_set = pkl.load(f)

    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set

    train_x = train_x.astype(theano.config.floatX)
    valid_x = valid_x.astype(theano.config.floatX)
    test_x = test_x.astype(theano.config.floatX)

    train_y = train_y.astype('int32')
    valid_y = valid_y.astype('int32')
    test_y = test_y.astype('int32')

    if args.verbose:
        print 'train x\t', train_x.shape
        print 'train y\t', train_y.shape
        print 'valid x\t', valid_x.shape
        print 'valid y\t', valid_y.shape
        print 'test x\t', test_x.shape
        print 'test y\t', test_y.shape

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def build_mlp(input_var=None):
    """
    Build a 2 layer MLP for MNIST with 1 linear output unit
    :param n_hidden: 
    :param input_var: 
    :return: 
    """

    softmax = lasagne.nonlinearities.softmax
    linear = lasagne.nonlinearities.linear

    net = lyr.InputLayer(shape=(None, 784), input_var=input_var)
    net = lyr.DenseLayer(net, num_units=800)
    # net = lyr.DenseLayer(net, num_units=800)
    net = lyr.DenseLayer(net, num_units=1, nonlinearity=linear)

    return net


def build_theano_fn():
    # Create network
    input_var = T.matrix('input_var')
    target_var = T.matrix('target_var')
    update_var = T.matrix('')

    network = build_mlp(input_var=input_var)
    params = lyr.get_all_params(network)
    output = lyr.get_output(network, inputs=input_var)

    # preds = T.argmax(output, axis=1)
    preds = output
    grad_pred_params = T.grad(output[0, 0], params)

    print 'Compiling functions...'
    predict_fn = theano.function(inputs=[input_var], outputs=preds)
    grad_fn = theano.function(inputs=[input_var], outputs=grad_pred_params)
    print 'compiled.'

    return predict_fn, grad_fn, network


def main():
    args = utils.get_args()

    # Settings
    BATCH_SIZE = 128
    NB_EPOCHS = args.epochs  # default 25
    LRN_RATE = 0.0001 / 28

    if args.verbose:
        PRINT_DELAY = 1
    else:
        PRINT_DELAY = 500

    # if running on server (MILA), copy dataset locally
    dataset_path = utils.init_dataset(args, 'mnist')

    # load dataset
    data = load_mnist_dataset(dataset_path, args)
    train_feats, train_targets, valid_feats, valid_targets, test_feats, test_targets = data

    # Get theano functions
    predict, preds_grad, network = build_theano_fn()

    print 'Starting training...'

    hist_errors = []
    running_err_avg = 0
    running_last_err_avg = 0

    # Create mask that will be used to create sequences
    mask = np.tril(np.ones((784, 784), dtype=theano.config.floatX), 0)

    for i in xrange(NB_EPOCHS):

        print 'Epoch #%s of %s' % ((i + 1), NB_EPOCHS)

        epoch_err = []
        num_batch = 0
        t_epoch = time.time()

        # iterate over minibatches for training
        schemes_train = ShuffledScheme(examples=train_feats.shape[0], batch_size=1)

        # We deal with 1 example as if it was an episode
        for batch_idx in schemes_train.get_request_iterator():

            batch_preds = []
            batch_error = []
            batch_grads = []

            num_batch += 1
            t_batch = time.time()

            train_x = train_feats[batch_idx]
            true_y = train_targets[batch_idx]

            nb_seq = 0
            for t in xrange(784):

                # apply mask at fixed interval, making sequences of pixels appear
                if (t + 1) % 28 == 0:
                    nb_seq += 1

                    seq_x = train_x * mask[t]
                    pred_y = predict(seq_x)
                    grad = preds_grad(seq_x)

                    batch_preds.append(pred_y[0, 0])
                    batch_grads.append(grad)

            # update params based on experience
            old_param_values = lyr.get_all_param_values(network)
            new_param_values = old_param_values

            for pred, grad in zip(batch_preds, batch_grads):

                error = (true_y - pred)
                delta_params = LRN_RATE * error * grad
                new_param_values += delta_params
                batch_error.append(error)

            lyr.set_all_param_values(network, new_param_values)

            last_error = np.abs(error[0])
            sqrd_error = np.linalg.norm(batch_error, 2)
            epoch_err.append(sqrd_error)
            running_err_avg = 0.05 * sqrd_error + 0.95 * running_err_avg
            running_last_err_avg = 0.05 * last_error + 0.95 * running_last_err_avg

            if num_batch % PRINT_DELAY == 0:
                print '- batch %s, err %s (avg %s), last %s (avg %s), in %s sec' % (num_batch,  np.round(sqrd_error, 4),
                                                                                        np.round(running_err_avg, 4),
                                                                                        np.round(last_error, 4),
                                                                                        np.round(running_last_err_avg, 4),
                                                                                        np.round(time.time() - t_batch, 2))

        print '- Epoch train (err %s) in %s sec' % (epoch_err, round(time.time() - t_epoch))

        # hist_errors.append(epoch_err)
        utils.dump_objects_output(args, epoch_err, 'epoch_%s_error_sqrd_norm.pkl' % (i + 1))

if __name__ == '__main__':
    main()