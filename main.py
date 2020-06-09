from fcn8_t import CNN
import matplotlib.pyplot as plt
import scipy.misc

#Defining variables
epochs = 60
batch_size=16
n_classes=2
dropout_coeff = 0.75

image_dir = "./data/test/input/umm_000075.png" 
image_shape=(160, 576)

model_dir = './model1'
weights_meta_dir = './model1/weights.meta'
output_dir = './data/test/output'

#Initialize fcn8 neural network
#cnn = CNN(epochs, batch_size, n_classes, dropout_coeff, image_shape)

#In order to run a prediction on an image after loading the model present in model_dir.
#prediction_image = cnn.predict(model_dir, weights_meta_dir, image_dir, image_shape, output_dir, 'output_adversary.png')

#In order to train the neural network and save the new model into model_dir
#cnn.run()

#In order to generate an adversarial image from a prediction
#adv_image = cnn.gen_adv(model_dir, weights_meta_dir, image_dir, image_shape, output_dir,  'noise.png')


#In the rest of the script we are going to attack an image from the testing set and compare the predictions between the original and the adversarial image.
original_image_dir = './data/test/input/umm_000075.png'
adversarial_image_dir = './data/test/output/imp_image.png'
original_image = scipy.misc.imresize(scipy.misc.imread(image_dir), image_shape)


#cnn1 = CNN(epochs, batch_size, n_classes, dropout_coeff, image_shape)
#original_prediction = cnn1.predict(model_dir, weights_meta_dir, original_image_dir, image_shape, output_dir, 'output_normal.png')
#print('image ok + pred')

#cnn2 = CNN(epochs, batch_size, n_classes, dropout_coeff, image_shape)
#adversarial_image = cnn2.gen_adv(model_dir, weights_meta_dir, original_image_dir, image_shape, output_dir,  'noised_image.png')
#print('adv image ok')
"""
cnn3 = CNN(epochs, batch_size, n_classes, dropout_coeff, image_shape)
adversarial_prediction = cnn3.predict(model_dir, weights_meta_dir, adversarial_image_dir, image_shape, output_dir, 'output_adversarial.png')
print('adv pred ok')

images = [original_image, original_prediction, adversarial_image, adversarial_prediction]
titles = ["Original image.", "Prediction on original image.", "Perturbed image.", "Prediction on perturbed image."]
fig = plt.figure(figsize=(2, 2))
columns = 2
rows = 2
for i in range(1, 5):
    fig.add_subplot(rows, columns, i)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()"""


cnn = CNN(epochs, batch_size, n_classes, dropout_coeff, image_shape)
#x = cnn.predict(model_dir, weights_meta_dir, adversarial_image_dir, image_shape, output_dir, 'output_adversarial_imp.png')
cnn.gen_gan_adv(model_dir, weights_meta_dir, original_image_dir, image_shape,'./data/data_road/training/gt_image_2/um_road_000028.png', output_dir,  'imp_image_1.png')
