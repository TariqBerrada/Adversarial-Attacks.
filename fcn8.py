import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import helper
from helper import read_labels, mean_squared_error

import os.path
import numpy as np
import cv2
import scipy.misc
from imputils import *

from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils

class CNN(object):
    
    def __init__(self, epochs, batch_size, n_classes, dropout_coeff, image_shape):
        """
        Initialization of the CNN class for the Semantic Segmentation model.
        """

        self.data_dir = './data'
        self.runs_dir = './runs'
        self.training_dir = './data/data_road/training'
        self.vgg_path = './model/vgg'
        self.model_dir = './data'
        self.model_directory = "./model1"
        self.model_filename = "FCN_8"
        self.signature_dir = "./model1/signature"

        self.sess = tf.InteractiveSession()

        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.dropout_coeff = dropout_coeff
        self.image_shape=(160, 576)

        #Defining placeholders for tensors that will be later used.
        self.label = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1], self.n_classes])
        self.learning_rate = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.get_batches_fn = helper.gen_batch_function(self.training_dir, self.image_shape)

        self.launched_layers = False

    def load_model(self, model_dir, weights_meta_dir):
        """
        Loads frozen inference graph and associated weights.
        """
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(weights_meta_dir)
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        print("Model loaded.")
            
        #graph = tf.get_default_graph()
    
    def predict(self, image_dir, img_shape, output_dir, output_name='output.png'):
        """
        Runs a prediction on an image.
        Returns the image overlayed with a green mask corresponding to pixels detected as 'road' class.
        """

        image_file = scipy.misc.imresize(scipy.misc.imread(image_dir), img_shape)
        
        print("Importing model.")
        graph = tf.get_default_graph()
        #image_input_1 = graph.get_tensor_by_name("image_input_1:0")
        #keep_prob_1 = graph.get_tensor_by_name("keep_prob_1:0")
        graph.get_operations()
        if not self.launched_layers:
            self.image_input, self.keep_prob, self.layer3, self.layer4, self.layer7 = self.load_vgg(self.sess, self.vgg_path)
            self.model_output = self.layers(self.layer3, self.layer4, self.layer7, self.n_classes)
            self.logits = graph.get_tensor_by_name("fcn_logits:0")
            self.logits = tf.reshape(self.logits, (-1, self.n_classes), name="fcn_logits")
        
        X_batch = [image_file.astype("float")]

        image_to_feed = np.array(X_batch, dtype="float")
        im_softmax = self.sess.run([tf.nn.softmax(self.logits)], {'keep_prob:0': 1.0, 'image_input:0': image_to_feed})    
        im_softmax = im_softmax[0][:, 1].reshape(img_shape[0], img_shape[1])
        im_pred = im_softmax
        segmentation = (im_softmax > 0.5).reshape(img_shape[0], img_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image_file)
        street_im.paste(mask, box=None, mask=mask)
        street_im = np.array(street_im)

        scipy.misc.imsave(output_dir + "/" + output_name, street_im)

        return street_im

    def gen_adv(self, image_dir, img_shape, output_dir='./data/test/output', output_name='output_adv.png', epsilon = 0.05):
        """
        Generates an adversarial example using FGSM.
        """
    
        image_file = scipy.misc.imresize(scipy.misc.imread(image_dir), img_shape)
 
        graph = tf.get_default_graph()
        self.logits = graph.get_tensor_by_name("fcn_logits:0")
        self.logits = tf.reshape(self.logits, (-1, self.n_classes), name="fcn_logits")
        image_input = graph.get_tensor_by_name("image_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.loss_obj = graph.get_tensor_by_name("fcn_loss_obj:0")
        self.loss = graph.get_tensor_by_name("fcn_loss_obj:0")

        print("Feeding image to model.")            
        image_to_feed = np.array([image_file], dtype="float")
        J = self.sess.run([tf.gradients([tf.nn.softmax(self.logits)], [image_input])], {'keep_prob:0': 1.0, 'image_input:0': image_to_feed})  
        
        print("Gradient calculated.")
        sign_J = np.sign(J)
        x = image_file/255
        adv_image = x + epsilon*sign_J
        adv_image = np.clip(adv_image, 0, 1)
        adv_image = np.reshape(adv_image, (img_shape[0], img_shape[1], 3))
        print("Adversary image generated.")
        print("Saving image to output folder.")
        scipy.misc.imsave(output_dir + "/" + output_name, adv_image)

        return adv_image     

    def gen_loss_p(self, image_file, img_shape, labels_dir):
        """
        Generates prediction loss for method 3.
        """
        graph = tf.get_default_graph()
        print("got graph")
        self.logits = graph.get_tensor_by_name("fcn_logits:0")
        self.logits = tf.reshape(self.logits, (-1, self.n_classes), name="fcn_logits")
        image_input = graph.get_tensor_by_name("image_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        print("Feeding image to model.")
        image_to_feed = np.array([scipy.misc.imresize(image_file, img_shape)], dtype="float")
        pred_mask_tensor = tf.nn.softmax(self.logits)[:,1]
        print("defined pred mask tensor")
        pred_mask = self.sess.run([tf.nn.softmax(self.logits)], {'keep_prob:0':1.0, 'image_input:0':image_to_feed})
        print("pred mask was run successfully")
        pred_mask=np.array(pred_mask)
        pred_mask = pred_mask[0][:, 1].reshape(img_shape[0], img_shape[1])

        real_mask_image = scipy.misc.imresize(read_labels(labels_dir), img_shape)
        print("shape of mask image", np.shape(real_mask_image))
        real_mask_tensor = tf.convert_to_tensor(real_mask_image)
        real_mask_tensor = tf.reshape(real_mask_tensor, shape = (-1, img_shape[0]*img_shape[1]))
        real_mask_tensor = tf.cast(real_mask_tensor, dtype = tf.float32)

        pred_loss = mean_squared_error(pred_mask_tensor, real_mask_tensor)
        pred_grad = tf.gradients(pred_loss, image_input)

        pred_loss_grad = self.sess.run(pred_grad, {'keep_prob:0':1.0, 'image_input:0':image_to_feed})
        pred_loss_grad = np.reshape(pred_loss_grad, (160, 576, 3))
        pred_loss_val = self.sess.run(pred_loss, {'keep_prob:0':1.0, 'image_input:0':image_to_feed})

        return pred_loss_val, pred_loss_grad

    def get_pred_step(self, image_file, img_shape, labels_dir, sess):
        
        graph = tf.get_default_graph()
        print("got graph")
        self.logits = graph.get_tensor_by_name("fcn_logits:0")
        self.logits = tf.reshape(self.logits, (-1, self.n_classes), name="fcn_logits")
        #image_input = graph.get_tensor_by_name("image_input:0")
        #keep_prob = graph.get_tensor_by_name("keep_prob:0")
        print("Feeding image to model.")
        image_to_feed = np.array([scipy.misc.imresize(image_file, img_shape)], dtype="float")
        pred_mask_tensor = tf.nn.softmax(self.logits)[:,1]
        print("defined pred mask tensor")
        pred_mask = self.sess.run([tf.nn.softmax(self.logits)], {'keep_prob:0':1.0, 'image_input:0':image_to_feed})
        print("pred mask was run successfully")
        pred_mask=np.array(pred_mask)
        pred_mask = pred_mask[0][:, 1].reshape(img_shape[0], img_shape[1])

        real_mask_image = scipy.misc.imresize(read_labels(labels_dir), img_shape)
        print("shape of mask image", np.shape(real_mask_image))
        real_mask_tensor = tf.convert_to_tensor(real_mask_image)
        real_mask_tensor = tf.reshape(real_mask_tensor, shape = (-1, img_shape[0]*img_shape[1]))
        real_mask_tensor = tf.cast(real_mask_tensor, dtype = tf.float32)


        pred_loss = mean_squared_error(pred_mask_tensor, real_mask_tensor)

        pred_grad = tf.gradients(pred_loss, image_input)

        pred_loss_grad = self.sess.run(pred_grad, {'keep_prob:0':1.0, 'image_input:0':image_to_feed})
        pred_loss_grad = np.reshape(pred_loss_grad, (160, 576, 3))
        pred_loss_val = self.sess.run(pred_loss, {'keep_prob:0':1.0, 'image_input:0':image_to_feed})

        return pred_loss_val, pred_loss_grad

    def gen_imp_adv(self, image_dir, img_shape, output_dir='./data/test/output', output_name='output_imp_adv.png', delta = 0.01, k = 100, n_iter =15, m = 4800, Dmax = np.inf):
        """
        Generates imperceptibl adversarial example.
        """
        
        image_file = scipy.misc.imresize(scipy.misc.imread(image_dir), img_shape)

        graph = tf.get_default_graph()
        self.logits = graph.get_tensor_by_name("fcn_logits:0")
        self.logits = tf.reshape(self.logits, (-1, self.n_classes), name="fcn_logits")
        image_input = graph.get_tensor_by_name("image_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.loss_obj = graph.get_tensor_by_name("fcn_loss_obj:0")
        self.loss = graph.get_tensor_by_name("fcn_loss_obj:0")
        print("Feeding image to model.")
        
        image_to_feed = np.array([image_file], dtype="float")
        print("Calculating gradient and prediction.")

        #Calculating sensibility matrix for the first iteration.
        iterator = 1
        distance = 0.0
        perturbations_mat = np.zeros((img_shape[0], img_shape[1]))
        while distance < Dmax and iterator <= n_iter:
            print("Iteration %d out of %d"%(iterator, n_iter))
            gap = tf.nn.softmax(self.logits) - (1/k)*tf.math.log(np.exp(1)*tf.math.exp(-k*tf.nn.softmax(self.logits)))
            gap_array = self.sess.run([tf.gradients([gap], [image_input])], {'keep_prob:0': 1.0, 'image_input:0': image_to_feed})  
            im_softmax = self.sess.run([tf.nn.softmax(self.logits)], {'keep_prob:0': 1.0, 'image_input:0': image_to_feed})
            
            gradients = self.sess.run([tf.gradients([tf.nn.softmax(self.logits)], [image_input])], {'keep_prob:0': 1.0, 'image_input:0': image_to_feed})  
            gradients = np.reshape(gradients, (img_shape[0], img_shape[1], 3))
            gap_array = np.reshape(gradients, (img_shape[0], img_shape[1], 3))
            gap_array = rgb2gray(gap_array)
            gap_array = im_softmax[0][:, 0] - im_softmax[0][:, 1]
            gap_array = np.reshape(gap_array, (img_shape[0], img_shape[1]))
            gap_array = np.ones(((img_shape[0], img_shape[1])))
            print("Calculating perturbation priority.")
            std = np.zeros(img_shape)
            image_gray = rgb2gray(image_to_feed[0])
            for i in range(1, img_shape[0]-1):
                for j in range(1, img_shape[1]-1):
                    std[i, j] = SD(image_gray, (i, j), 5)

            perturbation_priority = np.multiply(gap_array, std)
            #perturbation_priority[segmentation == False] = 0
            id_to_change = get_n_biggest(perturbation_priority, m)

            sign_J = np.sign(gradients)
            adv_image = np.reshape(image_to_feed, (img_shape[0], img_shape[1], 3))
            if iterator == 1:
                adv_image = adv_image/255
            
            for idx in id_to_change:
                adv_image[idx] = adv_image[idx] + delta*sign_J[idx]

            adv_image = np.clip(adv_image, 0, 1)
            adv_image = np.reshape(adv_image, (img_shape[0], img_shape[1], 3))
            image_to_feed = np.array([adv_image], dtype="float")

            sensibility_mat = 1/std
            sensibility_mat = np.clip(sensibility_mat, 0, 10000)
            sensibility_mat[sensibility_mat == 10000] = 0
            
            for idx in id_to_change:
                perturbations_mat[idx] += delta

            #perturbations_mat = np.clip(np.abs(adv_image - image_file/255), 0, 1)
            #perturbations_mat = rgb2gray(perturbations_mat)
            distance = distance_metric(image_file, sensibility_mat, perturbations_mat)
            print("D(X*, X) = %f"%distance)
            iterator += 1

        print("Adversary image generated.")
        print("Saving image to output folder.")
        scipy.misc.imsave(output_dir + "/" + output_name, adv_image)

        return adv_image   
    
    def run(self):
        """
        Does a training for the semantic segmentation algorithm.
        Predicts on the test folder then saves predictions to a corresponding folder. 
        """
        helper.maybe_download_pretrained_vgg(self.model_dir)

        #Importing VGG layers
        self.image_input, self.keep_prob, self.layer3, self.layer4, self.layer7 = self.load_vgg(self.sess, self.vgg_path)
        self.model_output = self.layers(self.layer3, self.layer4, self.layer7, self.n_classes)
        self.logits, self.train_op, self.cross_entropy_loss = self.optimize(self.model_output, self.label, self.learning_rate, self.n_classes)
        
        self.sess.run(tf.global_variables_initializer())
        print("Starting training.")
        self.train_nn(self.sess, self.epochs, self.batch_size, self.get_batches_fn, self.train_op, self.cross_entropy_loss, self.image_input, self.label, self.keep_prob, self.learning_rate)

        #Runnig the model on the test set and saving predictions.
        helper.save_inference_samples(self.runs_dir, self.data_dir, self.sess, self.image_shape, self.logits, self.keep_prob, self.image_input)

        print("Done.")

    def train_nn(self, sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, label, keep_prob, learning_rate):
        """
        Trains the semantic segmentation model.
        Saves the frozen inference graph and associated weights for further use.
        """
        self.keep_prob_value = .5
        self.learning_rate_value = 0.001
        for epoch in range(epochs):
            self.total_loss = 0.0
            print("Epoch %d out of %d."%(epoch+1, epochs))
            for X_batch, gt_batch in get_batches_fn(batch_size):    
                print("Importing a new batch for training.")
                self.loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={input_image:X_batch, label:gt_batch, keep_prob:self.keep_prob_value, learning_rate:self.learning_rate_value})
                self.total_loss += self.loss
            print("Loss = %s"%self.total_loss)
        
        print("Saving graph.")
        g = sess.graph
        gdef = g.as_graph_def()
        print("Saving model.")
        tf.io.write_graph(gdef, self.model_directory, "model_trained.pb", True)
        print("Initializing train saver.")
        self.saver = tf.train.Saver()
        print("saving weights")
        self.saver.save(sess, os.path.join(self.model_directory, "weights"))
        print("Session saved.")

    def optimize(self, nn_last_layer, labels, learning_rate, n_classes):
        """
        Model optimization using cross-entropy loss and AdamOptimizer.
        """

        self.logits = tf.reshape(nn_last_layer, (-1, n_classes), name="fcn_logits")
        self.label_reshaped = tf.reshape(labels, (-1, n_classes))
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.label_reshaped[:], name="fcn_loss_obj")
        self.loss = tf.reduce_mean(self.cross_entropy, name="fcn_loss")
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, name="fcn_train_op")

        return self.logits, self.train, self.loss        

    def layers(self, layer3_output, layer4_output, layer7_output, n_classes):
        """
        Defining layers that will be added to the VGG instead of the last dense layer.
        The dense layer is replaced with Conv2D transpose layers to recover original resolution for the mask.
        Skip layers are added in order to avoid overfitting.
        """

        self.fcn8 = tf.layers.conv2d(layer7_output, filters=n_classes, kernel_size=1, name="fcn8")
        self.fcn9 = tf.layers.conv2d_transpose(self.fcn8, filters=layer4_output.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='SAME', name='fcn9')
        self.fcn9_skip_connected = tf.add(self.fcn9, layer4_output, name="fcn9_plus_vgg_layer4")
        self.fcn10 = tf.layers.conv2d_transpose(self.fcn9_skip_connected, filters=layer3_output.get_shape().as_list()[-1], kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_plus_conv2D")
        self.fcn10_skip_connected = tf.add(self.fcn10, layer3_output, name="fcn10_plus_vgg_layer3")
        self.fcn11 = tf.layers.conv2d_transpose(self.fcn10_skip_connected, filters=n_classes, kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

        self.launched_layers = True

        return self.fcn11

    def load_vgg(self, sess, vgg_path):
        """
        Extracts the important VGG layers from the pre-trained model.
        """

        self.model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
        self.graph = tf.get_default_graph()
        self.image_input = self.graph.get_tensor_by_name('image_input:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
        self.layer3 = self.graph.get_tensor_by_name('layer3_out:0')
        self.layer4 = self.graph.get_tensor_by_name('layer4_out:0')
        self.layer7 = self.graph.get_tensor_by_name('layer7_out:0')

        return self.image_input, self.keep_prob, self.layer3, self.layer4, self.layer7
