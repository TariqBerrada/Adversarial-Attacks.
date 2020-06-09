import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import time


from IPython.display import Image, display

import PIL.Image
from fcn8 import *

import vgg16

vgg16.maybe_download()

def load_image(filename, max_size=None):
    """
    Loads image file from directory.
    """

    image = PIL.Image.open(filename)
    if max_size is not None:
        factor = max_size / np.max(image.size)
        size = np.array(image.size) * factor
        size = size.astype(int)
        image = image.resize(size, PIL.Image.LANCZOS)

    return np.float32(image)

def save_image(image, filename):
    """
    Saves image to directory.
    """

    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')

def plot_image_big(image):
    """
    Plot function.
    """

    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    display(PIL.Image.fromarray(image))

def plot_images(content_image, style_image, mixed_image):
    """
    Plot function, for notebook cells.
    """

    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    smooth = True
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")

    # Plot the style-image
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def mean_squared_error(a, b):
    """
    Calculates MSE between two tensors of the same shape.
    """
    return tf.reduce_mean(tf.square(a-b))

def content_loss(session, model, content_image, layer_ids):
    """
    Calculates the content loss by comparing the node values in different layers of the VGG16 when run on the original and generated images.
    """

    feed_dict = model.create_feed_dict(image=content_image)
    layers = model.get_layer_tensors(layer_ids)
    values = session.run(layers, feed_dict=feed_dict)

    with model.graph.as_default():
        layer_losses = []
        for value, layer in zip(values, layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(layer, value_const)
            layer_losses.append(loss)
        total_loss = tf.reduce_mean(layer_losses)

    return total_loss

def gram_matrix(tensor):
    """
    Measures the correlation between channels after flattening the filter images into vectors.
    """

    shape = tensor.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram

def denoise_loss(model):
    """
    Difference between the original image and the image translated by 1 pixel in the x and y direction.
    """

    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) +\
           tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))

    return loss

def image_change(content_image, mask_image, content_layer_ids, w_c=1.5, w_d=0.3, w_p=0.02, num_iter=20, step_size=10.0, epochs = 10,batch_size=16, n_classes=2, dropout_coeff = 0.75, image_shape=(160, 576)):
    content_image = scipy.misc.imresize(content_image, image_shape)

    print("shape of the content image: ", np.shape(content_image))

    model = vgg16.VGG16()
    session = tf.InteractiveSession(graph = model.graph)

    loss_content = content_loss(session = session, model = model, content_image = content_image, layer_ids = content_layer_ids)
    loss_denoise = denoise_loss(model)
    cnn = CNN(epochs, batch_size, n_classes, dropout_coeff, image_shape)
    cnn.load_model('./model1', './model1/weights.meta')
    loss_pred , _ = cnn.gen_loss_p(content_image, image_shape,'./data/data_road/training/gt_image_2/um_road_000028.png')

    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')
    adj_pred = tf.Variable(1e-10, name='adj_pred')

    session.run([adj_content.initializer, adj_denoise.initializer, adj_pred.initializer])

    update_adj_content = adj_content.assign(1.0/(loss_content+1e-10))
    update_adj_denoise = adj_denoise.assign(1.0/(loss_denoise+1e-10))
    update_adj_pred = adj_pred.assign(1.0/(loss_pred+1e-10))

    loss_combined = w_c*adj_content*loss_content + w_d*adj_denoise*loss_denoise

    gradient = tf.gradients(loss_combined, model.input)

    run_list = [gradient, update_adj_content, update_adj_denoise, update_adj_pred]
    mixed_image = np.random.rand(160, 576, 3) + 128

    for i in range(num_iter):
        print("iteration %d out of %d."%(i, num_iter))
        feed_dict = model.create_feed_dict(image=mixed_image)
        grad, adj_content_val, adj_denoise_val, adj_pred_val = session.run(run_list, feed_dict=feed_dict)
        grad_s = np.squeeze(grad)
        _ , grad_p = cnn.gen_loss_p(mixed_image, image_shape,'./data/data_road/training/gt_image_2/um_road_000028.png')
        grad = -w_p*grad_p + grad_s

        step_size_scaled = step_size/(np.std(grad) + 1e-8)

        mixed_image -= grad*step_size_scaled
        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        plt.imsave('image_gen_%d.jpg'%(i), mixed_image/255)

    return mixed_image/255

