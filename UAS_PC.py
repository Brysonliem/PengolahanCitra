# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Flatten, Dense

# Define input shape
input_shape = (256, 256, 3)

# Define generator model
def build_generator():
    # Input layer
    input_layer = Input(shape=input_shape)

    # Convolutional layer
    conv1 = Conv2D(64, (5, 5), strides=(1, 1), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = LeakyReLU(alpha=0.2)(bn1)

    # Convolutional layer
    conv2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = LeakyReLU(alpha=0.2)(bn2)

    # Convolutional layer
    conv3 = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(relu2)
    bn3 = BatchNormalization()(conv3)
    relu3 = LeakyReLU(alpha=0.2)(bn3)

    # Convolutional layer
    conv4 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(relu3)
    bn4 = BatchNormalization()(conv4)
    relu4 = LeakyReLU(alpha=0.2)(bn4)

    # Transposed convolutional layer
    deconv1 = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')(relu4)
    bn5 = BatchNormalization()(deconv1)
    relu5 = LeakyReLU(alpha=0.2)(bn5)

    # Transposed convolutional layer
    deconv2 = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(relu5)
    bn6 = BatchNormalization()(deconv2)
    relu6 = LeakyReLU(alpha=0.2)(bn6)

    # Transposed convolutional layer
    deconv3 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(relu6)
    bn7 = BatchNormalization()(deconv3)
    relu7 = LeakyReLU(alpha=0.2)(bn7)

    # Output layer
    output_layer = Conv2D(3, (5, 5), activation='tanh', padding='same')(relu7)

    # Define generator model
    generator = Model(input_layer, output_layer)

    return generator

# Define discriminator model
def build_discriminator():
    # Input layer
    input_layer = Input(shape=input_shape)

    # Convolutional layer
    conv1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input_layer)
    relu1 = LeakyReLU(alpha=0.2)(conv1)

    # Convolutional layer
    conv2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(relu1)
    relu2 = LeakyReLU(alpha=0.2)(conv2)

    # Convolutional layer
    conv3 = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(relu2)
    relu3 = LeakyReLU(alpha=0.2)(conv3)

    # Convolutional layer
    conv4 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(relu3)
    relu4 = LeakyReLU(alpha=0.2)(conv4)

    # Flatten layer
    flat = Flatten()(relu4)

    # Dense layer
    dense = Dense(1, activation='sigmoid')(flat)

    # Define discriminator model
    discriminator = Model(input_layer, dense)

    return discriminator

# Define CycleGAN model
def build_cyclegan():
    # Generate images from A to B
    generator_ab = build_generator()

    # Generate images from B to A
    generator_ba = build_generator()

    # Discriminator for images from A
    discriminator_a = build_discriminator()

    # Discriminator for images from B
    discriminator_b = build_discriminator()

    # Define loss functions
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Define loss functions for generator
    def generator_loss(real_image_a, generated_image_b, real_image_b, generated_image_a):
        # Loss for generator A to B
        loss_ab = loss_object(tf.ones_like(generated_image_b), generated_image_b)

        # Loss for generator B to A
        loss_ba = loss_object(tf.ones_like(generated_image_a), generated_image_a)

        # Cycle loss
        cycle_loss = loss_object(real_image_a, generated_image_a) + loss_object(real_image_b, generated_image_b)

        # Identity loss
        identity_loss_a = loss_object(real_image_a, generator_ba(real_image_a))
        identity_loss_b = loss_object(real_image_b, generator_ab(real_image_b))

# Example of calling the functions
generator = build_generator()
discriminator = build_discriminator()
cyclegan = build_cyclegan()

# Print model summaries
print("Generator Model Summary:")
generator.summary()

print("\nDiscriminator Model Summary:")
discriminator.summary()

print("\nCycleGAN Model Summary:")
cyclegan.summary()
