import numpy as np
import scipy.misc
import tensorflow as tf

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.image.resize(image, (224, 224))
    image = image[None, ...]
    return image

def deprocess(tensor):
    J = np.array(tensor)
    J = np.resize(J, (224, 224, 3))
    return J

def create_adversarial_pattern(input_image, input_label, model, loss_object):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

def gen_adv(image_path):
    #Importing vgg16 model
    model = tf.keras.applications.vgg16.VGG16()
    decode_predictions = tf.keras.applications.vgg16.decode_predictions
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    
    #Importing image
    image = scipy.misc.imread(image_path)
    image_probs = model.predict(np.resize(image, (1, 224, 224, 3)))
    image = preprocess(image)

    #Road label
    label_in = 908
    label = tf.one_hot(label_in, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    #Getting gradient
    perturbations = create_adversarial_pattern(image, label, model, loss_object)
    J = deprocess(perturbations)
    return J

def perturb_image(image_path, epsilon):
    J = gen_adv(image_path)
    image = np.array(scipy.misc.imread(image_path))
    adv_image = scipy.misc.imread(image_path) + epsilon*scipy.misc.imresize(J, np.shape(image))
    adv_image = np.clip(adv_image, a_min=0, a_max = 255)
    return adv_image



image_dir = "./data/test/input/uu_000085.png" 
adversarial_image = perturb_image(image_dir, 0.1)
im_per = perturb_image(image_dir, 0.1)
scipy.misc.imsave('./data/test/input/uu_000085_adv.png', im_per/255)
