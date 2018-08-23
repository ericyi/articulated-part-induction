import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PN_DIR = os.path.join(BASE_DIR,'pointnet2')
sys.path.append(os.path.join(PN_DIR,'utils'))
import tensorflow as tf
import numpy as np
import tf_util

class PNMEMCell(tf.nn.rnn_cell.RNNCell):
  # npoint: number of points
  # nfea: number of features
  # input: (batch_size, npoint, 3+nfea)
  # hidden state: (c, h) with shape (batch_size, npoint, nfea)
  def __init__(self, npoint, is_training, bn=False, bn_decay=None, reuse=None):
    super(PNMEMCell, self).__init__(_reuse=reuse)
    self._npoint = npoint
    self._bn = bn
    self._istraining = is_training
    self._bndecay = bn_decay
    self._statesize = tf.TensorShape([self._npoint])
    self._outputsize = tf.TensorShape([self._npoint])

  @property
  def state_size(self):
    return self._statesize

  @property
  def output_size(self):
    return (self._outputsize, self._outputsize, tf.TensorShape([1]))

  def call(self, x, state):
    # x: (batch_size, npoint, 3+npoint), indicating the grouping, prob
    # state: (batch_size, npoint), indicating whether the point has been selected, prob
    # (segpred, confpred): prob, logits
    #### use pn to compute score
    xyz, grouping = tf.split(x, [3, self._npoint], axis=2)
    l0_points = tf.concat((tf.expand_dims(grouping,3), tf.expand_dims(tf.tile(tf.expand_dims(state,1),(1,self._npoint,1)),3)),3) #(batch_size, npoint, npoint, 2)
    xyz = tf.tile(tf.expand_dims(xyz,1),(1,self._npoint,1,1))
    U = tf.concat((xyz, l0_points),3) #(batch_size, npoint, npoint, 5)
    mlp = [16,64,256]
    for i, num_out_channel in enumerate(mlp):
        U = tf_util.conv2d(U, num_out_channel, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=self._bn, is_training=self._istraining,
                                    scope='conv_stage1_%d'%(i), bn_decay=self._bndecay)
    U_glb = tf.reduce_max(U, 2, keep_dims=True)
    mlp2 = [128,32,32]
    for i, num_out_channel in enumerate(mlp2):
        U_glb = tf_util.conv2d(U_glb, num_out_channel, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=self._bn, is_training=self._istraining,
                                    scope='conv_stage2_%d'%(i), bn_decay=self._bndecay)

    confpred = tf.reduce_max(U_glb, 1, keep_dims=True)
    confpred = tf_util.conv2d(confpred, 32, [1,1],
                                padding='VALID', stride=[1,1],
                                bn=False, is_training=self._istraining,
                                scope='conv_stage2_conf_3', bn_decay=self._bndecay) #(batch_size, 1, 1, 32)
    confpred = tf_util.conv2d(confpred, 16, [1,1],
                                padding='VALID', stride=[1,1],
                                bn=False, is_training=self._istraining,
                                scope='conv_stage2_conf_4', bn_decay=self._bndecay) #(batch_size, 1, 1, 32)
    confpred = tf_util.conv2d(confpred, 1, [1,1],
                                padding='VALID', stride=[1,1],
                                activation_fn=None, scope='conv_stage2_conf_5') #(batch_size, 1, 1, 1)
    confpred = tf.squeeze(confpred,[2,3])

    segpred = tf_util.conv2d(U_glb, 32, [1,1],
                                padding='VALID', stride=[1,1],
                                bn=self._bn, is_training=self._istraining,
                                scope='conv_stage2_seg_3', bn_decay=self._bndecay) #(batch_size, npoint, 1, 32)
    segpred = tf_util.conv2d(segpred, 16, [1,1],
                                padding='VALID', stride=[1,1],
                                bn=self._bn, is_training=self._istraining,
                                scope='conv_stage2_seg_4', bn_decay=self._bndecay) #(batch_size, npoint, 1, 32)
    segpred = tf_util.conv2d(segpred, 1, [1,1],
                                padding='VALID', stride=[1,1],
                                activation_fn=None, scope='conv_stage2_seg_5') #(batch_size, npoint, 1, 1)
    segpred = tf.squeeze(tf.nn.softmax(segpred, dim=1),3)
    selectpred = tf.squeeze(segpred,2)
    segpred = tf.reduce_sum(tf.multiply(segpred, grouping),1) #(batch_size, npoint)
    state = tf.multiply(1-state, segpred)+state

    return (segpred, selectpred, confpred), state

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((8,5,128,131))
        cell = PNMEMCell(128, None)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)
        print(outputs, state)
