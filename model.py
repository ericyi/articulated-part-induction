import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PN_DIR = os.path.join(BASE_DIR,'pointnet2')
sys.path.append(os.path.join(PN_DIR,'utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_sa_module_msg
from tf_sampling import farthest_point_sample, gather_point
from pnmem_cell import PNMEMCell
from scipy.optimize import linear_sum_assignment

def placeholder_inputs(nmask=10, num_point=512):
    pcpair_pl = tf.placeholder(tf.float32, shape=(None, num_point, 6))
    flow_pl = tf.placeholder(tf.float32, shape=(None, num_point, 3))
    vismask_pl = tf.placeholder(tf.float32, shape=(None, num_point))
    momasks_pl = tf.placeholder(tf.float32, shape=(None, nmask, num_point))
    return pcpair_pl, flow_pl, vismask_pl, momasks_pl

def corrsfea_extractor(xyz, is_training, bn_decay, scopename, reuse, nfea=64):
    ############################
    # input
    #   xyz: (B x N x 3)
    # output
    #   corrsfea: (B x N x nfea)
    ############################
    num_point = xyz.get_shape()[1].value
    l0_xyz = xyz
    l0_points = l0_xyz
    with tf.variable_scope(scopename) as myscope:
        if reuse:
            myscope.reuse_variables()
        # Set Abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module_msg(l0_xyz, l0_points, 256, [0.1,0.2], [64,64], [[64,64],[64,64],[64,128]], is_training, bn_decay, scope='corrs_layer1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='corrs_layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, use_xyz=False, is_training=is_training, bn_decay=bn_decay, scope='corrs_layer3')
        # Feature Propagation layers
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='corrs_fa_layer1')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='corrs_fa_layer2')
        l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,64], is_training, bn_decay, scope='corrs_fa_layer3')
        # FC layers
        net = tf_util.conv1d(l0_points, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='corrs_fc1', bn_decay=bn_decay)
        net = tf_util.conv1d(net, nfea, 1, padding='VALID', activation_fn=None, scope='corrs_fc2')
        corrsfea = tf.reshape(net, [-1, num_point, nfea])
    return corrsfea

def corrs_flow_pred_net(xyz1, xyz2, net1, net2, scopename, reuse, is_training, bn_decay, nsmp=256, nfea=64):
    #########################################
    # input
    #   xyz1, xyz2: (B x N x 3)
    #   net1, net2: (B x N x nfea)
    # output
    #   pred_flow: (B x N x 3)
    #   pred_vismask: (B x N)
    #   fpsidx1, fpsidx2: (B x nsmp)
    #   matching_score_sub: (B x nsmp x nsmp)
    #########################################
    num_point = xyz1.get_shape()[1].value
    with tf.variable_scope(scopename) as myscope:
        if reuse:
            myscope.reuse_variables()
        # sub-sample to predict vismask and flow
        fpsidx1 = farthest_point_sample(nsmp, xyz1) # Bxnsmp
        idx = tf.where(tf.greater_equal(fpsidx1,0))
        fpsidx1 = tf.concat((tf.expand_dims(tf.cast(idx[:,0],tf.int32),-1),tf.reshape(fpsidx1,[-1,1])),1)
        xyz1_sub = tf.reshape(tf.gather_nd(xyz1, fpsidx1), [-1, nsmp, 3])
        net1_sub = tf.reshape(tf.gather_nd(net1, fpsidx1), [-1, nsmp, nfea])
        fpsidx2 = farthest_point_sample(nsmp, xyz2) # Bxnsmp
        idx = tf.where(tf.greater_equal(fpsidx2,0))
        fpsidx2 = tf.concat((tf.expand_dims(tf.cast(idx[:,0],tf.int32),-1),tf.reshape(fpsidx2,[-1,1])),1)
        xyz2_sub = tf.reshape(tf.gather_nd(xyz2, fpsidx2), [-1, nsmp, 3])
        net2_sub = tf.reshape(tf.gather_nd(net2, fpsidx2), [-1, nsmp, nfea])
        net_combined_sub = tf.concat((tf.tile(tf.expand_dims(net1_sub, 2),[1,1,nsmp,1]),tf.tile(tf.expand_dims(net2_sub, 1),[1,nsmp,1,1])),-1)
        
        mlp_maskpred = [128,128,128]
        for i, num_out_channel in enumerate(mlp_maskpred):
            net_combined_sub = tf_util.conv2d(net_combined_sub, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conv%d_maskpred'%(i), bn_decay=bn_decay) 
        pred_vismask_sub = tf.reduce_max(net_combined_sub, 2, keep_dims=True)
        mlp2_maskpred = [128,64,32]
        for i, num_out_channel in enumerate(mlp2_maskpred):
            pred_vismask_sub = tf_util.conv2d(pred_vismask_sub, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conv_post_%d_maskpred'%(i), bn_decay=bn_decay) 
        pred_vismask_sub = tf_util.conv2d(pred_vismask_sub, 1, [1,1],
                                        padding='VALID', stride=[1,1],
                                        scope='conv_out_maskpred', activation_fn=None)
        pred_vismask_sub = tf.squeeze(pred_vismask_sub, [2])
        pred_vismask = pointnet_fp_module(xyz1, xyz1_sub, None, pred_vismask_sub, [], tf.constant(True), None, scope='interp_layer')
        pred_vismask = tf.squeeze(pred_vismask,2) # B x nsmp
        pred_vismask_sub = tf.stop_gradient(tf.sigmoid(pred_vismask_sub)) # B x nsmp x 1

        mlp0 = [8]
        for i, num_out_channel in enumerate(mlp0):
            net_combined_sub = tf_util.conv2d(net_combined_sub, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conv_prev_%d'%(i), bn_decay=bn_decay)
        net_combined_sub = tf_util.conv2d(net_combined_sub, 1, [1,1],
                                    padding='VALID', stride=[1,1],
                                    scope='conv_prev_3', activation_fn=None)
        U = tf.nn.softmax(net_combined_sub, 2) # B x nsmp x nsmp x 1
        matching_score_sub = tf.squeeze(net_combined_sub, -1)

        #### mask prob
        U = tf.concat((tf.multiply(U, tf.expand_dims(pred_vismask_sub,2)), tf.expand_dims(xyz2_sub, 1)-tf.expand_dims(xyz1_sub, 2)),-1)

        mlp = [32,64,128,256]
        for i, num_out_channel in enumerate(mlp):
            U = tf_util.conv2d(U, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)
        U = tf.reduce_max(U, 2)
        l1_xyz = xyz1_sub

        #### mask energy
        l1_points = tf.concat((U, pred_vismask_sub), -1)

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='corrs_layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, use_xyz=False, is_training=is_training, bn_decay=bn_decay, scope='corrs_layer3')
        # Feature Propagation layers
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='corrs_fa_layer1')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='corrs_fa_layer2')
        l0_points = pointnet_fp_module(xyz1, l1_xyz, None, l1_points, [128,128,64], is_training, bn_decay, scope='corrs_fa_layer3')
        # FC layers
        net = tf_util.conv1d(l0_points, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='corrs_fc1', bn_decay=bn_decay)
        net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='corrs_fc2')
        pred_flow = tf.reshape(net, [-1, num_point, 3])
    return pred_flow, pred_vismask, fpsidx1, fpsidx2, matching_score_sub

def trans_pred_net(xyz, flow, scopename, reuse, is_training, bn_decay=None, nfea=12):
    #########################
    # input
    #   xyz: (B x N x 3)
    #   flow: (B x N x 3)
    # output
    #   pred_trans: (B x N x nfea)
    #########################
    num_point = xyz.get_shape()[1].value
    with tf.variable_scope(scopename) as myscope:
        if reuse:
            myscope.reuse_variables()
        l0_xyz = xyz
        l0_points = flow
        # Set Abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module_msg(l0_xyz, l0_points, 256, [0.1,0.2], [64,64], [[64,64],[64,64],[64,128]], is_training, bn_decay, scope='trans_layer1', centralize_points=True)
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='trans_layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, use_xyz=True, is_training=is_training, bn_decay=bn_decay, scope='trans_layer3')
        # Feature Propagation layers
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='trans_fa_layer1')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='trans_fa_layer2')
        l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,64], is_training, bn_decay, scope='trans_fa_layer3')
        # FC layers
        net = tf_util.conv1d(l0_points, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='trans_fc1', bn_decay=bn_decay)
        net = tf_util.conv1d(net, nfea, 1, padding='VALID', activation_fn=None, scope='trans_fc2')
        pred_trans = tf.reshape(net, [-1, num_point, nfea])
    return pred_trans

def grouping_pred_net(xyz, flow, trans, scopename, reuse, is_training, bn_decay=None, nsmp=128):
    ########################################
    # input
    #   xyz: (B x N x 3)
    #   flow: (B x N x 3)
    #   trans: (B x N x nfea)
    # output
    #   pred_grouping_sub: (B x nsmp x nsmp) - logits
    #   fpsidx: (B x nsmp)
    ########################################
    num_point = xyz.get_shape()[1].value
    nfea = trans.get_shape()[2].value
    with tf.variable_scope(scopename) as myscope:
        if reuse:
            myscope.reuse_variables()
        # Grouping
        fpsidx = farthest_point_sample(nsmp, xyz) # Bxnsmp
        idx = tf.where(tf.greater_equal(fpsidx,0))
        fpsidx = tf.concat((tf.expand_dims(tf.cast(idx[:,0],tf.int32),-1),tf.reshape(fpsidx,[-1,1])),1)
        xyz_sub = tf.reshape(tf.gather_nd(xyz, fpsidx), [-1, nsmp, 3])
        flow_sub = tf.reshape(tf.gather_nd(flow, fpsidx), [-1, nsmp, 3])
        pred_sub = tf.reshape(tf.gather_nd(trans, fpsidx), [-1, nsmp, nfea])
        Rs = tf.reshape(pred_sub[:,:,:9],[-1, nsmp, 3, 3])
        ts = tf.reshape(pred_sub[:,:,9:],[-1, nsmp, 1, 3])
        ppdist = tf.expand_dims(xyz_sub,1)-tf.expand_dims(xyz_sub,2) # B x nsmp x nsmp x 3
        ppdist = tf.matmul(ppdist,Rs)+ts+tf.expand_dims(flow_sub,2) # B x nsmp x nsmp x 3

        U = tf.concat((tf.tile(tf.expand_dims(xyz_sub,1),(1,nsmp,1,1)),ppdist-tf.expand_dims(flow_sub,1)),-1) # B x nsmp x nsmp x 6
        mlp = [16,64,512]
        for i, num_out_channel in enumerate(mlp):
            U = tf_util.conv2d(U, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conv_stage1_%d'%(i), bn_decay=bn_decay)
        U_glb = tf.reduce_max(U, 2, keep_dims=True)
        mlp2 = [256,256,256]
        for i, num_out_channel in enumerate(mlp2):
            U_glb = tf_util.conv2d(U_glb, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conv_stage2_%d'%(i), bn_decay=bn_decay)
        U_combined = tf.concat((tf.tile(U_glb,(1,1,nsmp,1)),U),-1)
        mlp3 = [256,64,16]
        for i, num_out_channel in enumerate(mlp3):
            U_combined = tf_util.conv2d(U_combined, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conv_stage3_%d'%(i), bn_decay=bn_decay)
        U_combined = tf_util.conv2d(U_combined, 1, [1,1],
                                    padding='VALID', stride=[1,1],
                                    scope='conv_stage3_3', activation_fn=None) # B x nsmp x nsmp x 1

        pred_grouping_sub = tf.squeeze(U_combined, -1) # B x nsmp x nsmp
    return pred_grouping_sub, fpsidx

def seg_pred_net(xyz, grouping_sub, fpsidx, scopename, reuse, is_training, bn_decay=None, nmask=10):
    ########################################
    # input
    #   xyz: (B x N x 3)
    #   grouping_sub: (B x nsmp x nsmp)
    #   fpsidx: (B x nsmp)
    # output
    #   pred_seg_sub: (B x nmask x nsmp) - prob
    #   pred_conf: (B x nmask x 1) - logits
    ########################################
    nsmp = grouping_sub.get_shape()[1].value
    with tf.variable_scope(scopename) as myscope:
        if reuse:
            myscope.reuse_variables()
        xyz_sub = tf.reshape(tf.gather_nd(xyz, fpsidx), [-1, nsmp, 3])
        segfea_sub = tf.nn.sigmoid(grouping_sub) # B x nsmp x nsmp
        segfea_sub = tf.concat((xyz_sub, segfea_sub), 2) # B x nsmp x 3+nsmp
        segfea_sub = tf.tile(tf.expand_dims(segfea_sub,1),[1,nmask,1,1])
        cell = PNMEMCell(nsmp, is_training, bn=False, bn_decay=None)
        outputs, state = tf.nn.dynamic_rnn(cell, segfea_sub, dtype=segfea_sub.dtype)
        pred_seg_sub, pred_select_sub, pred_conf = outputs # B x nmask x nsmp, _, B x nmask x 1    
    return pred_seg_sub, pred_conf

def flowloss(pred_flow, gt_flow):
    """ pred_flow: B x N x 3,
        gt_flow: B x N x 3 """
    loss_flow = tf.reduce_mean(tf.reduce_sum(tf.square(pred_flow-gt_flow),-1))
    return loss_flow

def matchingloss(matching_score_sub, fpsidx1, fpsidx2, xyz1, xyz2, gt_flow, gt_vismask, nsmp=256, nfea=64):
    """ matching_score_sub: B x nsmp x nsmp,
        fpsidx1, fpsidx2: B x nsmp,
        xyz1, xyz2, gt_flow: B x N x 3,
        gt_vismask: B x N """
    xyz1_sub = tf.reshape(tf.gather_nd(xyz1, fpsidx1), [-1, nsmp, 3])
    xyz2_sub = tf.reshape(tf.gather_nd(xyz2, fpsidx2), [-1, nsmp, 3])
    gt_flow_sub = tf.reshape(tf.gather_nd(gt_flow, fpsidx1), [-1, nsmp, 3])
    gt_vismask_sub = tf.reshape(tf.gather_nd(gt_vismask, fpsidx1), [-1, nsmp])
    gt_matching_labels = tf.argmin(tf.reduce_sum(tf.square(tf.expand_dims(xyz2_sub, 1)-tf.expand_dims(xyz1_sub+gt_flow_sub, 2)),-1),2)
    gt_matching_labels = tf.stop_gradient(gt_matching_labels)
    loss_matching = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = matching_score_sub, labels = gt_matching_labels)
    loss_matching = tf.multiply(gt_vismask_sub, loss_matching)
    loss_matching = tf.divide(tf.reduce_sum(loss_matching), tf.reduce_sum(gt_vismask_sub)+1e-8)
    return loss_matching

def transloss(pred_trans, fpsidx, nsmp, xyz, gt_flow, momasks):
    """ pred_trans: B x npoint x ntransfea,
        xyz: B x npoint x 3,
        gt_flow: B x npoint x 3,
        momasks: B x nmask x npoint """
    ntransfea = pred_trans.get_shape()[2].value
    nmask = momasks.get_shape()[1].value
    momasks = tf.transpose(momasks,perm=[0,2,1])
    xyz_sub = tf.reshape(tf.gather_nd(xyz, fpsidx), [-1, nsmp, 3])
    flow_sub = tf.reshape(tf.gather_nd(gt_flow, fpsidx), [-1, nsmp, 3])
    pred_sub = tf.reshape(tf.gather_nd(pred_trans, fpsidx), [-1, nsmp, ntransfea])
    momasks_sub = tf.reshape(tf.gather_nd(momasks, fpsidx), [-1, nsmp, nmask])
    momasks_sub_matrix = tf.reduce_sum(tf.multiply(tf.expand_dims(momasks_sub,2),tf.expand_dims(momasks_sub,1)),-1)
    momasks_sub_matrix_normalized = tf.divide(momasks_sub_matrix,tf.reduce_sum(momasks_sub_matrix,2,keep_dims=True)+1e-8)

    Rs = tf.reshape(pred_sub[:,:,:9],[-1, nsmp, 3, 3])
    ts = tf.reshape(pred_sub[:,:,9:],[-1, nsmp, 1, 3])
    ppdist = tf.expand_dims(xyz_sub,1)-tf.expand_dims(xyz_sub,2) # B x nsmp x nsmp x 3
    ppdist = tf.matmul(ppdist,Rs)+ts+tf.expand_dims(flow_sub,2) # B x nsmp x nsmp x 3
    loss_trans = ppdist-tf.tile(tf.expand_dims(flow_sub,1),[1,nsmp,1,1])
    loss_trans = tf.reduce_sum(tf.square(loss_trans),-1)
    loss_trans = tf.multiply(loss_trans,momasks_sub_matrix_normalized)
    loss_trans = tf.divide(tf.reduce_sum(loss_trans), tf.reduce_sum(momasks_sub_matrix_normalized)+1e-8)
    return loss_trans

def groupingloss(pred_grouping, fpsidx, nsmp, momasks):
    """ pred_grouping: B x nsmp x nsmp,
        xyz: B x npoint x 3,
        flow: B x npoint x 3,
        momasks: B x nmask x npoint """
    nmask = momasks.get_shape()[1].value
    momasks = tf.transpose(momasks,perm=[0,2,1])
    momasks_sub = tf.reshape(tf.gather_nd(momasks, fpsidx), [-1, nsmp, nmask])
    momasks_sub_matrix = tf.reduce_sum(tf.multiply(tf.expand_dims(momasks_sub,2),tf.expand_dims(momasks_sub,1)),-1)
    loss_grouping = tf.nn.sigmoid_cross_entropy_with_logits(labels=momasks_sub_matrix, logits=pred_grouping)
    loss_grouping = tf.reduce_mean(loss_grouping)
    return loss_grouping

def interp_grouping(xyz, pred_grouping, fpsidx, nsmp, scopename, reuse):
    """ xyz: B x N x 3,
        pred_grouping: B x nsmp x nsmp,
        fpsidx: B x nsmp """
    num_point = xyz.get_shape()[1].value
    with tf.variable_scope(scopename) as myscope:
        if reuse:
            myscope.reuse_variables()
        xyz_sub = tf.reshape(tf.gather_nd(xyz, fpsidx), [-1, nsmp, 3])
        # row interp
        xyz_aug1 = tf.tile(tf.expand_dims(xyz, 1),(1,nsmp,1,1))
        xyz_aug1 = tf.reshape(xyz_aug1,(-1, num_point, 3))
        xyz_sub_aug1 = tf.tile(tf.expand_dims(xyz_sub,1),(1,nsmp,1,1))
        xyz_sub_aug1 = tf.reshape(xyz_sub_aug1,(-1, nsmp, 3))
        U_combined = tf.reshape(pred_grouping, (-1, nsmp, 1))
        U_combined = pointnet_fp_module(xyz_aug1, xyz_sub_aug1, None, U_combined, [], tf.constant(True), None, scope='interp_layer_row')
        U_combined = tf.reshape(U_combined,(-1, nsmp, num_point, 1))
        U_combined = tf.transpose(U_combined, perm=(0,2,1,3))
        U_combined = tf.reshape(U_combined, (-1, nsmp, 1)) # B*npoint x nsmp x 1
        # column interp
        xyz_aug2 = tf.tile(tf.expand_dims(xyz, 1),(1,num_point,1,1))
        xyz_aug2 = tf.reshape(xyz_aug2,(-1, num_point, 3))
        xyz_sub_aug2 = tf.tile(tf.expand_dims(xyz_sub,1),(1,num_point,1,1))
        xyz_sub_aug2 = tf.reshape(xyz_sub_aug2,(-1, nsmp, 3))
        U_combined = pointnet_fp_module(xyz_aug2, xyz_sub_aug2, None, U_combined, [], tf.constant(True), None, scope='interp_layer_column')
        U_combined = tf.reshape(U_combined,(-1, num_point, num_point))
        U_combined = tf.transpose(U_combined, perm=(0,2,1))
    return U_combined

def hungarian_matching(pred_x, gt_x, curnmasks):
    """ pred_x, gt_x: B x nmask x nsmp
        curnmasks: B
        return matching_idx: B x nmask x 2 """
    batch_size = gt_x.shape[0]
    nmask = gt_x.shape[1]
    matching_score = np.matmul(gt_x,np.transpose(pred_x,axes=[0,2,1])) # B x nmask x nmask
    matching_score = 1-np.divide(matching_score, np.expand_dims(np.sum(pred_x,2),1)+np.sum(gt_x,2,keepdims=True)-matching_score+1e-8)
    matching_idx = np.zeros((batch_size, nmask, 2)).astype('int32')
    curnmasks = curnmasks.astype('int32')
    for i, curnmask in enumerate(curnmasks):
        row_ind, col_ind = linear_sum_assignment(matching_score[i,:curnmask,:curnmask])
        matching_idx[i,:curnmask,0] = row_ind
        matching_idx[i,:curnmask,1] = col_ind
    return matching_idx

def get_model_loss_stage1(pcpair, gt_vismask, gt_flow, momasks, is_training, bn_decay=None, nfea=64, ntransfea=12, nsmp1=256, nsmp2=128):
    ###############################################################
    # Train the correspondence proposal module and the flow module.
    # input
    #   pcpair: (B x N x 6)
    #   gt_vismask: (B x N)
    #   gt_flow: (B x N 3)
    #   momasks: (B x nmask x N)
    ###############################################################
    num_point = pcpair.get_shape()[1].value
    xyz1, xyz2 = tf.split(pcpair, [3, 3], axis=2)
    corrsfea = corrsfea_extractor(tf.concat((xyz1,xyz2),0), is_training, bn_decay, 'CorrsFeaExtractor', False, nfea=nfea)
    corrsfea1, corrsfea2  = tf.split(corrsfea, 2, 0)
    pred_flow, pred_vismask, fpsidx1, fpsidx2, matching_score_sub = corrs_flow_pred_net(xyz1, xyz2, corrsfea1, corrsfea2, 'CorrsFlowNet', False, is_training, bn_decay, nsmp=nsmp1, nfea=nfea)
    loss_flow = flowloss(pred_flow, gt_flow)
    loss_vismask = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_vismask, logits=pred_vismask))
    loss_matching = matchingloss(matching_score_sub, fpsidx1, fpsidx2, xyz1, xyz2, gt_flow, gt_vismask, nsmp=nsmp1, nfea=nfea)
    loss = loss_flow+loss_vismask+loss_matching
    return pred_flow, pred_vismask, loss_flow, loss_vismask, loss_matching, loss

def get_model_loss_stage2(pcpair, gt_vismask, gt_flow, momasks, is_training, bn_decay=None, nfea=64, ntransfea=12, nsmp1=256, nsmp2=128):
    ######################################################
    # Train the hypothesis generation and the verification 
    # submodule of the segmentation module.
    # input
    #   pcpair: (B x N x 6)
    #   gt_vismask: (B x N)
    #   gt_flow: (B x N 3)
    #   momasks: (B x nmask x N)
    ######################################################
    num_point = pcpair.get_shape()[1].value
    xyz = pcpair[:,:,:3]
    pred_trans = trans_pred_net(xyz, gt_flow, 'TransNet', False, is_training, bn_decay, nfea=ntransfea)
    pred_grouping_sub, fpsidx = grouping_pred_net(xyz, gt_flow, tf.stop_gradient(pred_trans), 'GroupingNet', False, is_training, bn_decay, nsmp=nsmp2)

    loss_trans = transloss(pred_trans, fpsidx, nsmp2, xyz, gt_flow, momasks)
    loss_grouping = groupingloss(pred_grouping_sub, fpsidx, nsmp2, momasks)
    loss = loss_trans+loss_grouping
    return pred_trans, pred_grouping_sub, loss_trans, loss_grouping, loss

def get_model_loss_stage3(pcpair, gt_vismask, gt_flow, momasks, is_training, bn_decay=None, nfea=64, ntransfea=12, nsmp1=256, nsmp2=128):
    #################################################
    # Train the hypothesis selection submodule of the
    # segmentation module.
    # input
    #   pcpair: (B x N x 6)
    #   gt_vismask: (B x N)
    #   gt_flow: (B x N 3)
    #   momasks: (B x nmask x N)
    #################################################
    num_point = pcpair.get_shape()[1].value
    xyz, xyz2 = tf.split(pcpair, [3, 3], axis=2)
    corrsfea = corrsfea_extractor(tf.concat((xyz,xyz2),0), tf.constant(False), None, 'CorrsFeaExtractor', False, nfea=nfea)
    corrsfea1, corrsfea2  = tf.split(corrsfea, 2, 0)
    pred_flow, _, _, _, _ = corrs_flow_pred_net(xyz, xyz2, corrsfea1, corrsfea2, 'CorrsFlowNet', False, tf.constant(False), None, nsmp=nsmp1, nfea=nfea)
    pred_flow = tf.stop_gradient(pred_flow)

    pred_trans = trans_pred_net(xyz, pred_flow, 'TransNet', False, tf.constant(False), None, nfea=ntransfea)
    pred_grouping_sub, fpsidx = grouping_pred_net(xyz, pred_flow, tf.stop_gradient(pred_trans), 'GroupingNet', False, tf.constant(False), None, nsmp=nsmp2)
    pred_grouping_sub = tf.stop_gradient(pred_grouping_sub)
    fpsidx = tf.stop_gradient(fpsidx)

    nsmp = pred_grouping_sub.get_shape()[1].value
    nmask = momasks.get_shape()[1].value
    xyz_sub = tf.reshape(tf.gather_nd(xyz[:,:,:3], fpsidx), [-1, nsmp, 3])
    momasks = tf.transpose(momasks,perm=[0,2,1])
    momasks_sub = tf.reshape(tf.gather_nd(momasks, fpsidx), [-1, nsmp, nmask])
    gt_conf_sub = tf.stop_gradient(tf.cast(tf.greater(tf.expand_dims(tf.reduce_sum(momasks_sub,1),-1),0),tf.float32)) # B x nmask x 1
    gt_seg_sub = tf.transpose(momasks_sub,perm=[0,2,1]) # B x nmask x nsmp

    pred_seg_sub, pred_conf = seg_pred_net(xyz, pred_grouping_sub, fpsidx, 'SegNet', False, is_training, bn_decay, nmask=nmask)
    matching_idx = tf.stop_gradient(tf.py_func(hungarian_matching, [pred_seg_sub, gt_seg_sub, tf.reduce_sum(gt_conf_sub,[1,2])], tf.int32)) # B x nmask x 2
    matching_idx_row = matching_idx[:,:,0]
    idx = tf.where(tf.greater_equal(matching_idx_row,0))
    matching_idx_row = tf.concat((tf.expand_dims(tf.cast(idx[:,0],tf.int32),-1),tf.reshape(matching_idx_row,[-1,1])),1)
    gt_seg_matched = tf.reshape(tf.gather_nd(gt_seg_sub, matching_idx_row), [-1, nmask, nsmp])
    matching_idx_column = matching_idx[:,:,1]
    idx = tf.where(tf.greater_equal(matching_idx_column,0))
    matching_idx_column = tf.concat((tf.expand_dims(tf.cast(idx[:,0],tf.int32),-1),tf.reshape(matching_idx_column,[-1,1])),1)
    pred_seg_matched = tf.reshape(tf.gather_nd(pred_seg_sub, matching_idx_column), [-1, nmask, nsmp])
    # comput iou
    matching_score = tf.reduce_sum(tf.multiply(gt_seg_matched, pred_seg_matched),2)
    negiou = -tf.divide(matching_score,tf.reduce_sum(gt_seg_matched,2)+tf.reduce_sum(pred_seg_matched,2)-matching_score+1e-8)
    negiou = tf.divide(tf.reduce_sum(tf.multiply(negiou, tf.squeeze(gt_conf_sub,2)),1), tf.reduce_sum(gt_conf_sub,[1,2])+1e-8) # B
    loss_seg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.squeeze(gt_conf_sub,2), logits=tf.squeeze(pred_conf,2)),1) # B
    loss_seg = loss_seg+negiou
    loss_seg = tf.reduce_mean(loss_seg)
    return pred_seg_sub, pred_conf, loss_seg

def eva_flow(pcpair, nfea=64, ntransfea=12, nsmp1=256, nsmp2=128):
    ###########################################################
    # Evaluate the deformation flow between a point cloud pair.
    # input
    #   pcpair: (B x N x 6)
    ###########################################################
    num_point = pcpair.get_shape()[1].value
    xyz, xyz2 = tf.split(pcpair, [3, 3], axis=2)
    corrsfea = corrsfea_extractor(tf.concat((xyz,xyz2),0), tf.constant(False), None, 'CorrsFeaExtractor', False, nfea=nfea)
    corrsfea1, corrsfea2  = tf.split(corrsfea, 2, 0)
    pred_flow, pred_vismask, _, _, _ = corrs_flow_pred_net(xyz, xyz2, corrsfea1, corrsfea2, 'CorrsFlowNet', False, tf.constant(False), None, nsmp=nsmp1, nfea=nfea)
    return pred_flow, pred_vismask

def eva_seg(pcpair, pred_flow, nfea=64, ntransfea=12, nsmp1=256, nsmp2=128, nmask=10):
    #####################################################
    # Evaluate the motion segmentation from a point cloud
    # equipped with a deformation flow.
    # input
    #   pcpair: (B x N x 6)
    #####################################################
    num_point = pcpair.get_shape()[1].value
    xyz, xyz2 = tf.split(pcpair, [3, 3], axis=2)
    pred_trans = trans_pred_net(xyz, pred_flow, 'TransNet', False, tf.constant(False), None, nfea=ntransfea)
    pred_grouping_sub, fpsidx = grouping_pred_net(xyz, pred_flow, pred_trans, 'GroupingNet', False, tf.constant(False), None, nsmp=nsmp2)
    pred_seg_sub, pred_conf = seg_pred_net(xyz, pred_grouping_sub, fpsidx, 'SegNet', False, tf.constant(False), None, nmask=nmask)
    pred_conf = tf.nn.sigmoid(pred_conf)
    xyz_sub = tf.reshape(tf.gather_nd(xyz, fpsidx), [-1, nsmp2, 3])
    #### up sample
    pred_seg = tf.transpose(pred_seg_sub, perm=[0,2,1]) # B x nsmp x nmask
    pred_seg = pointnet_fp_module(xyz, xyz_sub, None, pred_seg, [], tf.constant(True), None, scope='interp_layer_seg')
    pred_seg = tf.transpose(pred_seg, perm=[0,2,1]) # B x nmask x npoint
    return pred_seg, pred_conf

if __name__=='__main__':
    with tf.Graph().as_default():
        pcpair_inputs = tf.zeros((8,512,6))
        vismasks_inputs = tf.ones((8,512))
        flow_inputs = tf.ones((8,512,3))
        momasks_inputs = tf.ones((8,10,512))
        pred_flow, pred_vismask, loss_flow, loss_vismask, loss_matching, loss = get_model_loss_stage1(pcpair_inputs, vismasks_inputs, flow_inputs, momasks_inputs, tf.constant(True))
        print(pred_flow, pred_vismask, loss_flow, loss_vismask, loss_matching, loss)
        pred_trans, pred_grouping_sub, loss_trans, loss_grouping, loss = get_model_loss_stage2(pcpair_inputs, vismasks_inputs, flow_inputs, momasks_inputs, tf.constant(True))
        print(pred_trans, pred_grouping_sub, loss_trans, loss_grouping, loss)
        # pred_seg_sub, pred_conf, loss_seg = get_model_loss_stage3(pcpair_inputs, vismasks_inputs, flow_inputs, momasks_inputs, tf.constant(True))
        # print(pred_seg_sub, pred_conf, loss_seg)
        # pred_flow, pred_vismask = eva_flow(pcpair_inputs)
        # print(pred_flow, pred_vismask)
        # pred_seg, pred_conf = eva_seg(pcpair_inputs, pred_flow)
        # print(pred_seg, pred_conf)
