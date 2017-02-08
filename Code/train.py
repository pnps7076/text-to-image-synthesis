#!/usr/bin/python
import numpy as np
import tensorflow as tf
from DCGAN import DCGAN
from Data_Prep import Data_Prep
import cv2
import scipy.misc
def train_fn(nEpochs,image_path):
    '''Function to train DCGAN
        Input: nIter: Number of epochs to train DCGAN'''
    dcgan = DCGAN(batch_size=128,inp_size=4,output_size=64,z_dim=100,learning_rate=2e-4)
    dp = Data_Prep(image_path,batch_size=dcgan.batch_size,nImgs=8000,img_size=dcgan.output_size)
    loss,vars,outs = dcgan.build_model()
    for var in vars['vars_g']:
        print var.name,var.get_shape()
    for var in vars['vars_d']:
        print var.name,var.get_shape()
    d_optim = tf.train.AdamOptimizer(learning_rate=dcgan.learning_rate).minimize(loss['loss_d'],var_list=vars['vars_d'])
    g_optim = tf.train.AdamOptimizer(learning_rate=dcgan.learning_rate).minimize(loss['loss_g'],var_list=vars['vars_g'])
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    n = int(dp.nImgs*nEpochs/dcgan.batch_size)
    for i in range(n):
        real_imgs = dp.read_batch(overfit=False)
        z = np.random.uniform(-1,1,size=[dcgan.batch_size,dcgan.z_dim])
        _,curr_loss_d,d_real,d_fake,gen_imgs = sess.run([d_optim,loss['loss_d'],
                                                        outs['disc_real'],
                                                        outs['disc_fake'],
                                                        outs['gen_imgs']],
                                                        feed_dict={dcgan.z:z,
                                                        dcgan.real_images:real_imgs,
                                                        dcgan.train:True})
        _,curr_loss_g = sess.run([g_optim,loss['loss_g']],feed_dict={dcgan.z:z,dcgan.train:True})
        _,curr_loss_g = sess.run([g_optim,loss['loss_g']],feed_dict={dcgan.z:z,dcgan.train:True})
        if i%10 == 0:
            print "Iteration {0} Training loss:\nD_Loss:{1} \tG_Loss:{2}".format(i,curr_loss_d,curr_loss_g)
            gen_imgs = sess.run(dcgan.gen_imgs,feed_dict={dcgan.z:z,dcgan.train:False})
	    print "Mean real_D: {0} \t Mean fake_D: {1}".format(np.mean(d_real),np.mean(d_fake))
            print "Min output:{0} \t Max output:{1}".format(np.min(gen_imgs),np.max(gen_imgs))
            print "Mean generated:{0} \t Mean real:{1}".format(np.mean(gen_imgs),np.mean(real_imgs))
        if i%100==0:
            for j in range(10):
                scipy.misc.imsave('../../creaite_v2/Output/' + str(j) + '_' + str(i) + '_Gen.jpg',gen_imgs[j])
    saver.save(sess,'../ckpt/' + str(dcgan.batch_size) + '_' +str(nEpochs) + '.ckpt')

if __name__=="__main__":
    train_fn(nEpochs=20,image_path = '../../creaite_v2/images')
