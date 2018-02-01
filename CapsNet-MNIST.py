
# coding: utf-8

# # Capsule Networks
# 
# This is a Keras implementation of the CapsNet architecture proposed in https://arxiv.org/abs/1710.09829, to be submitted to the [Global NIPS implementation Challenge](https://nurture.ai/nips-challenge).

# In[1]:


from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, Dense, Flatten, Reshape, Dot, Multiply
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.engine.topology import Layer
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.activations import softmax
from keras import initializers, regularizers, constraints

from sklearn.model_selection import train_test_split

from keras import backend as K
import numpy as np

from os.path import exists, join
from os import mkdir

import shutil
import tensorflow as tf


# In[2]:


NUM_CLASSES = 10                       # Number of different classes (10 here for 0-9)
IMG_WIDTH = 28                         # Width of each image
IMG_HEIGHT = 28                        # Height of each image
NUM_CHANNELS = 1                       # Number of channels in each image (1 here, for grayscale)
BATCH_SIZE = 128                       # Batch size to be used while training
EPOCHS = 500                           # Number of epochs to train the model for
RECONSTRUCTION_REG = 0.0005            # Regularization factor multiplied with the reconstruction loss
LEARNING_RATE = 1e-3                   # Initial Learning rate used while training


# ## Model

# In[3]:


class Squash(Layer):
    
    def __init__(self, **kwargs):
        super(Squash, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Squash, self).build(input_shape)
        
    def call(self, x):
        x_norm = K.expand_dims(Norm()(x))
        return (K.square(x_norm) / (1 + K.square(x_norm))) * x / (x_norm + K.epsilon())


# In[4]:


class PrimaryCaps(Layer):
    
    def __init__(self, num_sharing_capsules=32, num_channels_per_capsule=8, kernel_size=9, **kwargs):
        self.num_sharing_capsules = num_sharing_capsules
        self.num_channels_per_capsule = num_channels_per_capsule
        self.kernel_size = kernel_size
        super(PrimaryCaps, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(PrimaryCaps, self).build(input_shape)
        
    def call(self, x):
        primary_caps_conv_out = Conv2D(filters=self.num_sharing_capsules * self.num_channels_per_capsule,
                                       kernel_size=self.kernel_size,
                                       strides=(2, 2))(x)
        primary_caps_out = Reshape((-1, self.num_channels_per_capsule))(primary_caps_conv_out)                                       
        return Squash()(primary_caps_out)
    
    def compute_output_shape(self, input_shape):
        img_size = input_shape[1]
        out_size = (img_size - self.kernel_size + 1) / 2
        return (input_shape[0], out_size * out_size * self.num_sharing_capsules, 
                self.num_channels_per_capsule)
    
    def get_config(self):
        config = {
            'num_sharing_capsules': self.num_sharing_capsules,
            'num_channels_per_capsule': self.num_channels_per_capsule,
            'kernel_size': self.kernel_size
        }
        base_config = super(PrimaryCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[5]:


class DigitCaps(Layer):
    
    
    def __init__(self, batch_size, routing_iter=3, num_capsules=NUM_CLASSES, 
                 num_channels_per_capsule=16,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        
        self.batch_size = batch_size
        self.routing_iter = routing_iter
        self.num_capsules = num_capsules
        self.num_channels_per_capsule = num_channels_per_capsule
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        
        super(DigitCaps, self).__init__(**kwargs)
    
    
    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.num_capsules, input_shape[1],
                                        input_shape[2], 
                                        self.num_channels_per_capsule), 
                                 name='W',
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        
        super(DigitCaps, self).build(input_shape)
         
        
    def call(self, x):
        x_shape = x.get_shape().as_list()
        b = K.zeros([self.batch_size, self.num_capsules, x_shape[1]])
        b = Input(tensor=b)
        x = K.repeat_elements(K.expand_dims(x, axis=1), self.num_capsules, axis=1)
        
        x_hat = K.map_fn(lambda i: K.batch_dot(i, self.W, axes=0), elems=x)
        
        for i in range(self.routing_iter):
            c = K.expand_dims(softmax(b, axis=-1))
            s = Multiply()([x_hat, c])
            s = K.sum(s, axis=2)
            v = Squash()(s)
            
            v_expanded = K.repeat_elements(K.expand_dims(v, 2), x_shape[1], axis=2)
            update = Multiply()([v_expanded, x_hat])
            update = K.sum(update, axis=-1)
            b += update
            
        return v


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsules, self.num_channels_per_capsule)
    
    
    def get_config(self):
        config = {
            'routing_iter': self.routing_iter,
            'num_capsules': self.num_capsules,
            'num_channels_per_capsule': self.num_channels_per_capsule,
        }
        base_config = super(DigitCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[33]:


class Mask(Layer):
    
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Mask, self).build(input_shape)
        
    def call(self, x):
        if isinstance(x, list):
            x, mask = x
            
        else:
            norm = Norm()(x)
            
            mask = K.argmax(norm)
            mask = K.one_hot(mask, num_classes=x.get_shape().as_list()[1])
            
        masked_digit_cap = K.batch_flatten(x * K.expand_dims(mask))            
        return masked_digit_cap
    
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1] * input_shape[0][-1])
        
        return (input_shape[0], input_shape[1] * input_shape[-1])


# In[34]:


class Decoder(Layer):
    
    def __init__(self, shape, **kwargs):
        self.shape = shape
        super(Decoder, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Decoder, self).build(input_shape)
        
    def call(self, x):
        dense1_out = Dense(512, activation='relu', name='decoder_dense1')(x)
        dense2_out = Dense(1024, activation='relu', name='decoder_dense2')(dense1_out)
        dense3_out = Dense(784, activation='sigmoid', name='decoder_final')(dense2_out)
        dense3_out = Reshape(self.shape)(dense3_out)
        
        return dense3_out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.shape[0], self.shape[1], self.shape[2])
    
    def get_config(self):
        config = {
            'shape': self.shape,
        }
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[35]:


class Norm(Layer):
    
    def __init__(self, **kwargs):
        super(Norm, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Norm, self).build(input_shape)
        
    def call(self, x):
        return K.sqrt(K.sum(K.square(x), axis=-1))
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


# In[66]:

def custom_margin_loss(y_true, y_pred):
    m_plus = 0.9
    m_minus = 0.1
    down_weigh = 0.5
    loss = y_true * K.square(K.maximum(0., m_plus - y_pred)) + down_weigh * (1 - y_true) * K.square(K.maximum(0., y_pred - m_minus))
    return K.mean(K.sum(loss, axis=-1))


class CapsNet(object):
    
    def __init__(self, x_train, y_train, x_test, y_test, routing_iter=3, 
                 learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                 rec_reg=RECONSTRUCTION_REG, log_dir='./tb_logs/', checkpoint_dir='./weights'):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.routing_iter = routing_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.rec_reg = rec_reg
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        if exists(log_dir):
            shutil.rmtree(log_dir)
            shutil.rmtree(checkpoint_dir)
        
        mkdir(log_dir)
        mkdir(checkpoint_dir)
            
    
        

    def build_model(self):
        input_shape = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
        x = Input(shape=input_shape)
        y = Input(shape=(NUM_CLASSES,))
        
        conv_out = Conv2D(filters=256, kernel_size=9, activation='relu', name='conv')(x)
        primary_caps_out = PrimaryCaps(name='primary_caps')(conv_out)
        digit_caps_out = DigitCaps(batch_size=self.batch_size, name='digit_caps')(primary_caps_out)
        
        masked_digit_cap = Mask(name='mask_true')([digit_caps_out, y])
        masked_pred_digit_cap = Mask(name='mask_pred')(digit_caps_out)
        
        reconstructed_out = Decoder(shape=input_shape)(masked_digit_cap)
        pred_reconstructed_out = Decoder(shape=input_shape)(masked_pred_digit_cap)
        pred_prob = Norm(name='pred_prob')(digit_caps_out)
        
        train_model = Model(inputs=[x, y], outputs=[pred_prob, reconstructed_out])
        pred_model = Model(inputs=[x, y], outputs=[pred_prob, pred_reconstructed_out])
        
        train_model.compile(optimizer=Adam(lr=self.learning_rate), 
                      loss=[custom_margin_loss, 'mse'], 
                      loss_weights=[1, self.rec_reg],
                      metrics={'pred_prob': 'accuracy'})
        
        pred_model.compile(optimizer=Adam(lr=self.learning_rate), 
                           loss=[custom_margin_loss, 'mse'],
                           loss_weights=[1, self.rec_reg],
                           metrics={'pred_prob': 'accuracy'})
        
        return train_model, pred_model
    
    
    def data_generator(self, x, y, mode='train', fraction_shifted=0.1):    
        if(mode == 'train'):
            data_gen = ImageDataGenerator(rescale=1/255., height_shift_range=fraction_shifted, 
                                          width_shift_range=fraction_shifted,
                                          fill_mode='constant',
                                          cval=0)
        else:
            data_gen = ImageDataGenerator(1/255.)
            
        data_gen = data_gen.flow(x, y, batch_size=self.batch_size)
        while(True):
            x, y = data_gen.next()
            if (len(x) < self.batch_size):
            	continue

            yield([x, y], [y, x])
    
    
    def train(self, model):

        tb = TensorBoard(log_dir=self.log_dir, histogram_freq=0,
                         write_graph=True, write_images=False)

        checkpointer = ModelCheckpoint(join(self.checkpoint_dir, 'chkpts.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                       monitor='val_loss', save_best_only=True, 
                                       save_weights_only=False, verbose=1)
        print('Fitting model...')
        
        x_train, x_valid, y_train, y_valid = train_test_split(self.x_train, self.y_train,
                                                              test_size=0.2, random_state=0)
        
        train_datagen = self.data_generator(x_train, y_train)
        train_steps = len(x_train) // self.batch_size
        valid_datagen = self.data_generator(x_valid, y_valid, mode='valid')
        valid_steps = len(x_valid) // self.batch_size
        
        model.fit_generator(train_datagen,
                            steps_per_epoch=train_steps, 
                            epochs=self.epochs, 
                            verbose=1,
                            validation_data=valid_datagen,
                            validation_steps=valid_steps,
                            callbacks=[checkpointer, tb])
        
        _, _, test_acc = model.evaluate([self.x_test, self.y_test], [self.y_test, self.x_test])
        print('Test accuracy: %.4f' % test_acc)
        print('Done.')


def load_data():

	# Data Preprocessing

	(x_train, y_train), (x_test, y_test) = mnist.load_data()


	x_train.shape  # Input should be a 4-D array, so we need to reshape this

	x_train.dtype  # The input type should be 'float' 

	y_train.shape  # The labels should be one-hot encoded, so we need to convert this

	# Reshaping the 3-D input images to 4-D and converting their data types to np.float32
	x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype(np.float32)
	x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype(np.float32)


	# Making the labels one-hot encoded
	y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
	y_test = to_categorical(y_test, num_classes=NUM_CLASSES)


# Training

if __name__ == '__main__':

	K.clear_session()
	caps_net = CapsNet(x_train, y_train, x_test, y_test)
	train_model, pred_model = caps_net.build_model()
	print(pred_model.summary())


	caps_net.train(model=train_model)

