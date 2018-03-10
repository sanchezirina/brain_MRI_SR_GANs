import tensorflow as tf
import numpy as np
import math
from datasetnoAD import Train_dataset
import nibabel as nib
import os

batch_size = 1
num_patches = 8
div_patches = 1

DEFAULT_SAVE_PATH_PREDICTIONS = '/work/isanchez/predictions/autoencoder/RMSProp2'
DEFAULT_SAVE_PATH_CHECKPOINTS = '/work/isanchez/g/autoencoder/RMSprop2/step'


def lrelu(x):
    return tf.maximum(x, 0.3 * x)


def autoencoder(input_shape=[None, 128, 128, 92, 1], n_filters=[1, 20, 20, 20], filter_sizes=[3, 3, 3, 3]):
    """Build a deep denoising autoencoder w/ tied weights.
    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    Raises
    ------
    ValueError
        Description
    """
    # %%
    # input to the network
    x = tf.placeholder(tf.float32, input_shape, name='x')
    current_input = x

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[4]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv3d(
                current_input, W, strides=[1, 2, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[3]]))
        output = lrelu(tf.add(
            tf.nn.conv3d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3], shape[4]]),
                strides=[1, 2, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x))

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost}


# %%
def train():
    traindataset = Train_dataset(batch_size=1)
    batch_size = 1
    num_patches = 8
    div_patches = 1
    iterations_train = int(math.ceil((len(traindataset.subject_list) * 1) / batch_size))  # entreno con todo
    n_epochs = 100

    # %%
    ae = autoencoder()
    learning_rate = 0.001
    # global_step = tf.Variable(0, trainable=False)
    # lr_v = tf.Variable(0.01, trainable=False)
    # decay_rate = 0.5
    # decay_steps = 1875  # every 3 epochs
    # learning_rate = tf.train.inverse_time_decay(lr_v, global_step=global_step, decay_rate=decay_rate,
    #                                             decay_steps=decay_steps)
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data

    step = 0
    for epoch_i in range(n_epochs):
        for i in range(0, iterations_train):
            ###====================== LOAD DATA ===========================###
            xt_total = traindataset.patches_true(i)
            for k in range(0, div_patches):
                xt = xt_total[k * int((batch_size * num_patches) / div_patches):(int(
                    (batch_size * num_patches) / div_patches) * k) + int(
                    (batch_size * num_patches) / div_patches)]
                # NORMALIZING
                for t in range(0, xt.shape[0]):
                    normfactor = (np.amax(xt[t])) / 2
                    if normfactor != 0:
                        xt[t] = ((xt[t] - normfactor) / normfactor)
                sess.run(optimizer, feed_dict={ae['x']: xt})
            print(epoch_i, i, sess.run(ae['cost'], feed_dict={ae['x']: xt}))
        if iterations_train % 20:
            y = sess.run(ae['y'], feed_dict={ae['x']: xt})
            if normfactor != 0:
                y_pred_img = ((y[2] + 1) * normfactor)
            img_pred = nib.Nifti1Image(y_pred_img, np.eye(4))
            img_pred.to_filename(os.path.join(DEFAULT_SAVE_PATH_PREDICTIONS, str(epoch_i) + str(i) + 'pred.nii.gz'))
            img = nib.Nifti1Image(((xt[2] + 1) * normfactor), np.eye(4))
            img.to_filename(os.path.join(DEFAULT_SAVE_PATH_PREDICTIONS, str(epoch_i) + str(i) + 'real.nii.gz'))

        if epoch_i + 1 % 20:
            saver = tf.train.Saver()
            saver.save(sess=sess, save_path=DEFAULT_SAVE_PATH_CHECKPOINTS, global_step=step)
            print("Saved step: [%2d]" % step)
            step = step + 1


# %%
if __name__ == '__main__':
    train()
