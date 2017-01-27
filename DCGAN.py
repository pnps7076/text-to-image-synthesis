#!/usr/bin/python
import numpy as np
import tensorflow as tf
import cv2
from Data_Prep import Data_Prep
from Utils.ops import conv2d,conv2d_transpose,Linear,lrelu

"""Class file for DCGAN that generates images from catalog images of backpacks,
skirts etc. Generator is a network composed of successive upconvolution layers.
Discriminator is a network composed of multiple convolution layers which ends
in a softmax layer.
"""
class DCGAN:
    def __init__(self,batch_size,inp_size,output_size,z_dim,learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.inp_size = inp_size
        self.output_size = output_size
        self.z_dim = z_dim
        self.loss_G = None
        self.loss_D = None
        self.optim_G = None
        self.optim_D = None
        self.z = None
        self.convs = []
        self.upconvs = []
        self.create_placeholders()

    def print_upconvs(self):
        print "Generator shapes"
        for u in self.upconvs:
            print u.get_shape()

    def print_convs(self):
        print "Discriminator shapes"
        for c in self.convs:
            print c.get_shape()

    def create_placeholders(self):
        """Function to create placeholders for DCGAN model."""
        self.z = tf.placeholder(tf.float32,shape = [self.batch_size,self.z_dim],name='inp_noise') #placeholder for input noise to generator
        self.fake_images = tf.placeholder(tf.float32,shape = [self.batch_size,self.output_size,self.output_size,3],name='fake_img') #feed fake image to discriminator
        self.train = tf.placeholder(tf.bool,shape=[],name='train_indicator') #to use for batchnorm
        self.real_images = tf.placeholder(tf.float32,shape = [self.batch_size,self.output_size,self.output_size,3],name='real_img') #feed real image to discriminator

    def create_generator(self,train=True):
        """Function to create generator and store its weights in G_weights
        Returns:
            g6_out: A batch of tensors of shape [batch_size,s,s,3] which are the generated images
        """
        g1_inp_shape = (self.inp_size**2)*512
        with tf.variable_scope('Generator') as outer_scope:
            self.g_inp = Linear(self.z,g1_inp_shape,name='Noise2G',train=train)
            self.g_inp = tf.reshape(self.g_inp,[self.batch_size,self.inp_size,self.inp_size,512])
            self.g1_out = conv2d_transpose(self.g_inp,
                                            output_shape=[self.batch_size,s8,s8,256],
                                            name='Gen1',
                                            train=train)
            self.g2_out = conv2d_transpose(self.g1_out,
                                            output_shape=[self.batch_size,s4,s4,128],
                                            name='Gen2',
                                            train=train)
            self.g3_out = conv2d_transpose(self.g2_out,
                                            output_shape=[self.batch_size,s2,s2,64],
                                            name='Gen3',
                                            train=train)
            self.g4_out = conv2d_transpose(self.g3_out,
                                            output_shape = [self.batch_size,s,s,3],
                                            name='Gen4',train=False,
                                            act=False)
            self.upconvs = [self.g_inp,self.g1_out,self.g2_out,self.g3_out,self.g4_out]
            self.gen_imgs = (tf.nn.tanh(self.g4_out)/2. + 0.5)
            return (tf.nn.tanh(self.g4_out)/2. + 0.5)

    def create_discriminator(self,X,train=True,reuse=False):
        with tf.variable_scope('Discriminator',reuse = reuse) as outer_scope:
            self.conv1_out = conv2d(X,64,train=train,name='Disc1')
            self.conv2_out = conv2d(self.conv1_out,128,train=train,name='Disc2')
            self.conv3_out = conv2d(self.conv2_out,256,train=train,name='Disc3')
            self.conv4_out = conv2d(self.conv3_out,512,train=train,name='Disc4')
            n,h,w,c = self.conv4_out.get_shape()
            self.conv4_out_reshaped = tf.reshape(self.conv4_out,[self.batch_size,-1])
            self.disc_out = Linear(self.conv4_out_reshaped,1,train=False,name='Output',act=False)
            self.convs = [self.conv1_out,self.conv2_out,self.conv3_out,self.conv4_out,self.disc_out]
            return self.disc_out,tf.nn.sigmoid(self.disc_out)

    def build_model(self):
        gen_imgs = self.create_generator()
        disc_real,_disc_real = self.create_discriminator(self.real_images)
        disc_fake,_disc_fake = self.create_discriminator(gen_imgs,reuse = True)
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake,tf.ones_like(disc_fake)))
        disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real,tf.ones_like(disc_real)))
        disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake,tf.zeros_like(disc_fake)))
        disc_loss = disc_loss_fake + disc_loss_real
        d_vars = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'Generator' in var.name]
        loss = {
            'loss_d':disc_loss,
            'loss_g':gen_loss,
            'loss_d_real':disc_loss_real,
            'loss_d_fake':disc_loss_fake
        }
        vars = {
            'vars_d': d_vars,
            'vars_g': g_vars
        }
        outputs = {
            'disc_real': _disc_real,
            'disc_fake': _disc_fake,
            'gen_imgs': gen_imgs
        }
        self.print_upconvs()
        self.print_convs()
        return loss,vars,outputs


    def create_loss(self):
        self.reuse = None
        self.gener_out = self.create_generator(self.train,self.reuse)
        self.adv_pos = tf.reshape(self.create_discriminator(self.d_inp,self.train,self.reuse),[self.batch_size,1])
        self.reuse = True
        self.adv_neg = tf.reshape(self.create_discriminator(self.gener_out,self.train,self.reuse),[self.batch_size,1])
        self.loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.adv_neg,tf.ones_like(self.adv_neg)))
        self.loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.adv_pos,tf.ones_like(self.adv_pos)))
        self.loss_D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.adv_neg,tf.zeros_like(self.adv_pos)))
        self.loss_D = self.loss_D_real + self.loss_D_fake
        print "____Loss op created____"

    def create_optim(self):
        self.optim_G = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_G)
        self.optim_D = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_D)
        print "____Optim op created____"

if __name__ == "__main__":
    dgn = DCGAN(batch_size=128, inp_size=4, output_size=64, z_dim=100, learning_rate=0.0007)
