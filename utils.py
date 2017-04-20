import os
import glob
import cPickle as pkl
import shutil
import numpy as np
# import theano
# import lasagne.layers as lyr
import argparse
from distutils.dir_util import copy_tree


def init_dataset(args, dataset_name):
    """
    If running from MILA, copy on /Tmp/lacaillp/datasets/
    ---
    returns path of local dataset
    """
    if args.mila:
        src_dir = '/data/lisatmp3/lacaillp/datasets/'
        dst_dir = '/Tmp/lacaillp/datasets/'
    elif args.laptop:
        src_dir = '/Users/phil/datasets/'
        dst_dir = src_dir
    else:
        raise 'Location entered not valid (MILA/laptop)'

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if not os.path.exists(dst_dir + dataset_name):
        print 'Dataset not stored locally, copying %s to %s...' \
            % (dataset_name, dst_dir)
        shutil.copytree(src_dir + dataset_name, dst_dir + dataset_name, )
        print 'Copy completed.'

    return dst_dir + dataset_name


def dump_objects_output(args, object, filename):
    """
    Dumps any object using pickle into ./output/objects_dump/ directory
    """
    if args.mila:
        path = '/Tmp/lacaillp/output/objects_dump/'
    elif args.laptop:
        path = '/Users/phil/output/objects_dump/'

    if not os.path.exists(path):
        os.makedirs(path)

    full_path = os.path.join(path, filename)
    with open(full_path, 'wb') as f:
        pkl.dump(object, f)
    print 'Object saved to file %s' % filename


def save_model(args, network, filename):
    """
    Saves the parameters of a model to a pkl file
    Will try to get save it in './output/saved_models', otherwise will create it
    """
    if args.mila:
        path = '/Tmp/lacaillp/output/saved_models/'
    elif args.laptop:
        path = '/Users/phil/output/saved_models/'

    if not os.path.exists(path):
        os.makedirs(path)

    full_path = os.path.join(path, filename)
    with open(full_path, 'wb') as f:
        pkl.dump(lyr.get_all_param_values(network), f)
    print 'Model saved to file %s' % filename


def reload_model(args, network, filename, rslts_src):
    """
    Returns the network loaded of the parameters
    Will try to get filename in './output/saved_models'
    """
    if args.mila:
        src_dir = os.path.join('/data/lisatmp3/lacaillp/results/', str(rslts_src), 'saved_models/')
        dst_dir = '/Tmp/lacaillp/input/saved_models/'
    elif args.laptop:
        src_dir = '/Users/phil/input/saved_models/'
        dst_dir = src_dir

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    full_path = os.path.join(dst_dir, filename)

    print 'Copying saved model %s locally...' % filename
    shutil.copy(os.path.join(src_dir, filename), full_path)
    print 'Completed.'

    print 'Initiating loading...'
    try:
        with open(full_path, 'rb') as f:
            values = pkl.load(f)
    except:
        print 'An error occured, model wasn\'t loaded.'
        return False
    else:
        lyr.set_all_param_values(network, values)
        print 'Network was successfully loaded from %s' % full_path
        return True


def move_results_from_local():
    """
    Copy results stored on Tmp/lacaillp/output to Tmp/lacaillp/lisatmp3.
    Used at end of run. If successful, deletes the local results
    """

    src_dir = '/Tmp/lacaillp/output/'
    dst_dir = '/data/lisatmp3/lacaillp/output/'

    if not os.path.exists(src_dir):
        print '%s doesn\'t exist, nothing was copied.' % src_dir
    else:

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        try:
            copy_tree(src_dir, dst_dir)
        except:
            print 'Copy of data wasn\'t successful, local data was not deleted.'
        else:
            print 'Copy of data to %s was successful, local copy will be deleted...' % dst_dir
            shutil.rmtree(src_dir)
            print 'Local data was deleted from %s' % src_dir


def convert_to_sequence(data_x):
    """
    Converts mnist array to matrix representing sequence of pixels
    :param data_x: numpy array to be converted [nb examples, dimensions] 
    :return: seq_x: numpy array converted to sequence [nb examples, time, dimensions]
    """

    dims = data_x.shape[1]
    seq_x = np.zeros(shape=(data_x.shape[0], dims, dims))

    for t in xrange(dims):

        if t > 0:
            seq_x[:, t] = seq_x[:, t - 1]
        seq_x[:, t, t] = data_x[:, t]

    return seq_x


def get_args():
    """
    Returns the arguments passed by command-line
    """
    parser = argparse.ArgumentParser()
    load_src = parser.add_mutually_exclusive_group(required=True)
    load_src.add_argument('-m', '--mila', help='If running from MILA servers',
                          action='store_true')
    load_src.add_argument('-l', '--laptop', help='If running from laptop',
                          action='store_true')

    parser.add_argument('-e', '--epochs', help='Max number of epochs for training',
                        type=int, default=25)
    parser.add_argument('-g', '--gen', help='Number of images to generate',
                        type=int, default=5)
    parser.add_argument('-v', '--verbose', help='High verbose option used for debug or dev',
                        action='store_true')
    parser.add_argument('-s', '--save', help='Nb of epochs between saving model',
                        type=int, default=0)
    parser.add_argument('-r', '--reload', help='Reload previously trained model (rslts_src, id)',
                        type=str, default=None, nargs='+')

    return parser.parse_args()
