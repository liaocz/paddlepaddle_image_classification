import paddle.v2 as paddle
import numpy as np
import os
import sys
# for inference
from PIL import Image

with_gpu = os.getenv("WITH_GPU", '0') != '0'

def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts, num_channels = None):
        #image Convolution group used for vgg network
        return paddle.networks.img_conv_group(
            input=ipt, num_channels=num_channels, pool_size=2,
            pool_stride=2, conv_num_filter=[num_filter] * groups,
            conv_filter_size=3, conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True, conv_batchnorm_drop_rate=dropouts,
            pool_type=paddle.pooling.Max())

    #define five convolution block
    #convolution kernel 3 * 3, pooling size 2 * 2
    conv1 = conv_block(input, 64, 2, [0.3, 0], 3)
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    #define dropouts
    drop = paddle.layer.dropout(input = conv5, dropout_rate = 0.5)
    #define fully-connected
    fc1 = paddle.layer.fc(input = drop, size = 512, act = paddle.activation.Linear())
    #define batch normalization
    bn = paddle.layer.batch_norm(input = fc1, act = paddle.activation.Relu(),
                layer_attr = paddle.attr.Extra(drop_rate = 0.5))
    fc2 = paddle.layer.fc(input = bn, size = 512, act = paddle.activation.Linear())

    return fc2

def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    # The storage order of the loaded image is W(widht),
    # H(height), C(channel). PaddlePaddle requires
    # the CHW order, so transpose them.
    im = im.transpose((2, 0, 1))  # CHW
    # In the training phase, the channel order of CIFAR
    # image is B(Blue), G(green), R(Red). But PIL open
    # image in RGB mode. It must swap the channel order.
    im = im[(2, 1, 0), :, :]  # BGR
    im = im.flatten()
    im = im / 255.0
    return im

def main():
    #paddlepaddle init and using gpu
    paddle.init(use_gpu = with_gpu, trainer_count = 1)

    #input data
    #cifrar10 32 * 32 and 3 channels
    datadim = 3 * 32 * 32
    classdim = 10

    #data_layer
    image = paddle.layer.data(name = "image", type = paddle.data_type.dense_vector(datadim))

    #configure vgg network
    net = vgg_bn_drop(image)

    #define classifier using Softmax
    out = paddle.layer.fc(input = net, size = classdim, act = paddle.activation.Softmax())

    #define cost function and output
    lb1 = paddle.layer.data(name = "label", type = paddle.data_type.integer_value(classdim))
    cost = paddle.layer.classification_cost(input = out, label = lb1)

    ######  training model  ######
    #creating parameters
    parameters = paddle.parameters.create(cost)

    #create optimizer
    momentum_optimizer = paddle.optimizer.Momentum(
        momentum = 0.9, regularization = paddle.optimizer.L2Regularization(rate = 0.0002 * 128),
        learning_rate = 0.1 / 128.0, learning_rate_decay_a = 0.1, learning_rate_decay_b = 50000 * 100,
        learning_rate_schedule = 'discexp')

    #create trainer
    trainer = paddle.trainer.SGD(
        cost = cost, parameters = parameters, update_equation = momentum_optimizer)

    # End batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)

            result = trainer.test(
                reader = paddle.batch(
                    paddle.dataset.cifar.test10(), batch_size = 128),
                feeding = {'image': 0,
                         'label': 1})
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
            
    ########    training    ######
    trainer.train(
        reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.cifar.train10(), buf_size = 50000),
            batch_size = 128),
        num_passes = 200,
        event_handler = event_handler,
        feeding = {'image': 0,
                 'label': 1})

    test_data = []
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    test_data.append((load_image(cur_dir + '/image/dog.png'), ))

    # users can remove the comments and change the model name
    # with open('params_pass_50.tar', 'r') as f:
    #    parameters = paddle.parameters.Parameters.from_tar(f)

    probs = paddle.infer(
        output_layer = out, parameters = parameters, input = test_data)
    lab = np.argsort(-probs)  # probs and lab are the results of one batch data
    print "Label of image/dog.png is: %d" % lab[0][0]

if __name__ == '__main__':
    main()
