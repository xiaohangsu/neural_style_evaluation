caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()

import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import tempfile
import argparse

niter = 0  # number of iterations to train
batch_size = 64


# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

# Download just a small subset of the data for this exercise.
# To download the entire dataset, set `full_dataset = True`.
full_dataset = True
if full_dataset:
    NUM_STYLE_IMAGES = -1
    NUM_STYLE_LABELS = 20

else:
    NUM_STYLE_IMAGES = 71267
    NUM_STYLE_LABELS = 20


import os
weights = caffe_root + 'models/finetune_flickr_style/weights.pretrained.caffemodel'
weights2 = caffe_root + 'models/finetune_flickr_style/weights.scratch.caffemodel'
assert os.path.exists(weights)

# # Load ImageNet labels to imagenet_labels
# imagenet_label_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
# imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))
# assert len(imagenet_labels) == 1000
# print('Loaded ImageNet labels:\n', '\n'.join(imagenet_labels[:10] + ['...']))

# Load style labels to style_labels
style_label_file = caffe_root + 'examples/finetune_flickr_style/style_names.txt'
style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
if NUM_STYLE_LABELS > 0:
    style_labels = style_labels[:NUM_STYLE_LABELS]
print('\nLoaded style labels:\n', ', '.join(style_labels))




from caffe import layers as L
from caffe import params as P

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data, label=None, train=True, num_classes=1000, classifier_name='fc8', learn_all=False):
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name



def style_net(train=True, learn_all=False, subset=None):
    if subset is None:
        subset = 'train' if train else 'test'
    source = caffe_root + 'data/flickr_style/'+ subset + '.txt'
    transform_param = dict(mirror=train, crop_size=227,
        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=batch_size, new_height=256, new_width=256, ntop=2)
    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=NUM_STYLE_LABELS,
                    classifier_name='fc8_flickr',
                    learn_all=learn_all)



def disp_preds(net, image, labels, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    print('top %d predicted %s labels =' % (k, name))
    print('\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k)))

def disp_style_preds(net, image):
    disp_preds(net, image, style_labels, name='style')



from caffe.proto import caffe_pb2

def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.01
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = caffe_root + 'models/finetune_flickr_style'
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(str(s))
        return f.name

def run_solvers(niter, solvers, disp_interval=1):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print('%3d) %s' % (it, loss_disp))     
    # Save the learned weights from both nets.
    #weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join('/home/xiaohangsu/caffe/models/finetune_flickr_style', filename)
        print(weights[name])
        s.net.save(weights[name])
    return loss, acc, weights



def front_train():
    # Reset style_solver as before.
    style_solver_filename = solver(style_net(train=True))
    style_solver = caffe.get_solver(style_solver_filename)
    style_solver.net.copy_from(weights)

    # For reference, we also create a solver that isn't initialized from
    # the pretrained ImageNet weights.
    scratch_style_solver_filename = solver(style_net(train=True))
    scratch_style_solver = caffe.get_solver(scratch_style_solver_filename)
    scratch_style_solver.net.copy_from(weights2)

    print('Running solvers for %d iterations...' % niter)
    solvers = [('pretrained', style_solver),
               ('scratch', scratch_style_solver)]
    loss, acc, firstWeights = run_solvers(niter, solvers)
    print('Done.')

    train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
    train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
    style_weights, scratch_style_weights = firstWeights['pretrained'], firstWeights['scratch']

    # Delete solvers to save memory.
    del style_solver, scratch_style_solver, solvers


def eval_style_net(weights, test_iters=1):
    test_net = caffe.Net(style_net(train=False), weights, caffe.TEST)
    accuracy = 0
    for it in range(test_iters):
        accuracy += test_net.forward()['acc']
    accuracy /= test_iters
    return test_net, accuracy



def end_to_end_train(netName, weights):
    end_to_end_net = style_net(train=True, learn_all=True)

    # Set base_lr to 1e-3, the same as last time when learning only the classifier.
    # You may want to play around with different values of this or other
    # optimization parameters when fine-tuning.  For example, if learning diverges
    # (e.g., the loss gets very large or goes to infinity/NaN), you should try
    # decreasing base_lr (e.g., to 1e-4, then 1e-5, etc., until you find a value
    # for which learning does not diverge).
    base_lr = 0.001

    style_solver_filename = solver(end_to_end_net, base_lr=base_lr)
    style_solver = caffe.get_solver(style_solver_filename)
    style_solver.net.copy_from(weights)

    print('Running solvers for %d iterations...' % niter)
    solvers = [(netName, style_solver)]
    loss, acc, finetuned_weights = run_solvers(niter, solvers)
    print('Done.')

    # Delete solvers to save memory.
    del style_solver, solvers

    train_loss = loss[netName]
    train_acc = acc[netName]
    style_weights = finetuned_weights[netName]

    plt.plot(np.vstack([train_loss]).T)
    xlabel('Iteration #')
    ylabel('Loss')
    plt.show()

    plt.plot(np.vstack([train_acc]).T)
    xlabel('Iteration #')
    ylabel('Accuracy')
    plt.show()

def test(images):
    test_net, accuracy = eval_style_net(weights)
    print('Accuracy, trained from ImageNet initialization: %3.1f%%' % (100*accuracy, ))
    scratch_test_net, scratch_accuracy = eval_style_net(weights2)
    print('Accuracy, trained from random initialization: %3.1f%%' % (100*scratch_accuracy, ))


    for i in range(len(images)):
        image = test_net.blobs['data'].data[i]
        # print(test_net.blobs['label'][i]);
        # print('actual label =', style_labels[int(test_net.blobs['label'])])
        print("image: " + images[i])
        disp_style_preds(test_net, image)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='finetune_flickr_style Script')

    parser.add_argument(
        '-i', '--iter', type=int, default=0,
        help='Iteration Training Time')

    parser.add_argument(
        '-b', '--batch', type=int, default=64,
        help='Batch Image Size')

    parser.add_argument(
        '-f', '--front_train', action='store_true',
        help='before End to End Train')

    parser.add_argument(
        '-ei', '--end_to_end_train_imageNet', action='store_true',
        help='End to End Train Image_Net')

    parser.add_argument(
        '-es', '--end_to_end_train_scratch', action='store_true',
        help='End to End Train_Scratch')

    parser.add_argument(
        '-t', '--test', action='store_true',
        help='Test model')

    args = parser.parse_args()
    niter = args.iter  # number of iterations to train
    batch_size = args.batch

    if args.front_train:
        print("Start Front Training")
        front_train()

    if args.end_to_end_train_imageNet:
        print("Start End to End Training ImageNet")

        end_to_end_train("pretrained", weights)

    if args.end_to_end_train_scratch:
        print("Start End to End Training Scratch")
        end_to_end_train("scratch", weights2)

    if args.test:
        print("Start Testing")
        test_txt_lines = open(caffe_root +
            'data/flickr_style/test.txt', 'r').readlines()
        images = []
        for t in test_txt_lines:
            images.append(t.split(" ")[0].split("/")[-1])
            
        test(images)