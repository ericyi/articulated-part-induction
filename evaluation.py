import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import scipy.io as sio
import copy
from sklearn.neighbors import NearestNeighbors
from scipy.misc import comb
import data_prep
import seg_util


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--model_path', default='log3/best_model.ckpt', help='model checkpoint file path [default: log3/best_model.ckpt]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
NPOINT = 512
NMASK = 10

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(MODEL_DIR, FLAGS.model+'.py')
MODEL_PATH = FLAGS.model_path

HOSTNAME = socket.gethostname()

TEST_DATASET = data_prep.SynTestDataset('data/sf2f_test.mat', npoint=NPOINT)
ISFULL_MATCHING = True

# TEST_DATASET = data_prep.SynTestDataset('data/sf2p_test.mat', npoint=NPOINT)
# ISFULL_MATCHING = False

def evaluation():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pcpair_pl, flow_pl, _, _ = MODEL.placeholder_inputs(NMASK, NPOINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            print("--- Get model and loss")
            pred_flow, _ = MODEL.eva_flow(pcpair_pl)
            pred_trans, pred_grouping, pred_seg, pred_conf = MODEL.eva_seg(pcpair_pl, flow_pl, nmask=NMASK)
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, MODEL_PATH)

        ops = {'pcpair_pl': pcpair_pl,
               'flow_pl': flow_pl,
               'is_training_pl': is_training_pl,
               'pred_flow': pred_flow,
               'pred_trans': pred_trans,
               'pred_grouping': pred_grouping,
               'pred_seg': pred_seg,
               'pred_conf': pred_conf,
               'merged': merged}

        eval(sess, ops)

def get_batch_data(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_pcpair = np.zeros((bsize, NPOINT, 6), dtype=np.float)
    batch_flow = np.zeros((bsize, NPOINT, 3), dtype=np.float)
    batch_seg = np.zeros((bsize, NPOINT), dtype=np.int)
    for i in range(bsize):
        pc1, pc2, flow12, seg1 = dataset[idxs[i+start_idx]]
        batch_pcpair[i,:,:] = np.concatenate((pc1,pc2),1)
        batch_flow[i,:,:] = flow12
        batch_seg[i,:] = seg1
    return batch_pcpair, batch_flow, batch_seg

def rand_index_score(label_pred, label_gt):
    tp_plus_fp = comb(np.bincount(label_pred), 2).sum()
    tp_plus_fn = comb(np.bincount(label_gt), 2).sum()
    A = np.c_[(label_pred, label_gt)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(label_pred))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)
        
def eval(sess, ops):
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_ins = np.int(len(TEST_DATASET))
    RI_prev_all = list()
    RI_all = list()
    EPE_all = list()

    for ins_idx in range(0,num_ins,5):
        print(ins_idx)
        start_idx = ins_idx
        end_idx = ins_idx+1
        batch_pcpair, batch_flow, batch_seg = get_batch_data(TEST_DATASET, test_idxs, start_idx, end_idx)
        pc1_ini = copy.deepcopy(batch_pcpair[0,:,:3])
        pc1 = batch_pcpair[0,:,:3]
        pc2 = batch_pcpair[0,:,3:6]
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pc1)
        dd, idx = nbrs.kneighbors(pc1)
        dist_th = np.max(dd[:,1])
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pc2)
        dd, idx = nbrs.kneighbors(pc2)
        dist_th = np.maximum(dist_th, np.max(dd[:,1]))
        dist_th = np.minimum(dist_th, 0.06)

        #### step1 fit global motion
        feed_dict = {ops['pcpair_pl']: batch_pcpair,
                     ops['is_training_pl']: is_training}
        flow_pred = sess.run(ops['pred_flow'], feed_dict=feed_dict)
        pc1 = batch_pcpair[0,:,:3]
        pc_pred = pc1+flow_pred[0,...]
        ## globally align everything
        R, t = seg_util.fit_motion(pc1, pc_pred)
        batch_pcpair[0,:,:3] = np.matmul(pc1,R)+t
        feed_dict = {ops['pcpair_pl']: batch_pcpair,
                     ops['is_training_pl']: is_training}
        flow_pred = sess.run(ops['pred_flow'], feed_dict=feed_dict)
        pc1 = batch_pcpair[0,:,:3]
        pc_pred = pc1+flow_pred[0,...]
        ## globally align everything
        R, t = seg_util.fit_motion(pc1, pc_pred)
        batch_pcpair[0,:,:3] = np.matmul(pc1,R)+t
        feed_dict = {ops['pcpair_pl']: batch_pcpair,
                     ops['is_training_pl']: is_training}
        flow_pred = sess.run(ops['pred_flow'], feed_dict=feed_dict)
        pc1 = batch_pcpair[0,:,:3]
        pc_pred = pc1+flow_pred[0,...]

        #### iterative corrs and seg
        expand_eps_list = [0.2]+[0.15]*4+[0.1]*4
        for i, expand_eps in enumerate(expand_eps_list):
            ## seg
            batch_pcpair = np.concatenate((np.expand_dims(pc1,0),np.expand_dims(pc2,0)),2)
            feed_dict = {ops['pcpair_pl']: batch_pcpair,
                         ops['flow_pl']: np.expand_dims(pc_pred-pc1,0),
                         ops['is_training_pl']: is_training}
            trans_pred, grouping_pred, seg_pred, conf_pred = sess.run([
                ops['pred_trans'], ops['pred_grouping'], ops['pred_seg'], ops['pred_conf']], feed_dict=feed_dict)
            Rmodes, tmodes, nmodes, segidx, segidx2 = seg_util.decode_motion_modes(pc1, pc_pred-pc1, pc2, trans_pred[0], grouping_pred[0], seg_pred[0], conf_pred[0], eps=dist_th)
            subpc1, subpc2, attmask1, attmask2, segidx, segidx2, distmatrix, distmatrix2 = seg_util.gen_attention_mask(pc1, pc_pred, pc2, segidx, segidx2, Rmodes, tmodes, nmodes, NPOINT, expand_eps=expand_eps)
            for j in range(nmodes):
                subpc1[j,:,:] = np.matmul(subpc1[j,:,:], Rmodes[j,:,:])+tmodes[[j],:]

            ## corrs
            batch_pcpair = np.concatenate((subpc1, subpc2),-1)
            feed_dict = {ops['pcpair_pl']: batch_pcpair,
                         ops['is_training_pl']: is_training}
            flow_pred = sess.run(ops['pred_flow'], feed_dict=feed_dict)
            pc_pred, segidx = seg_util.motion_modes_aggreg_watt(subpc1, subpc1+flow_pred, attmask1)

        if ISFULL_MATCHING:
            pc_pred = pc2[np.argmin(np.sum((np.expand_dims(pc_pred,1)-np.expand_dims(pc2,0))**2,2),1),:]

        batch_pcpair = np.concatenate((np.expand_dims(pc1,0),np.expand_dims(pc2,0)),2)
        feed_dict = {ops['pcpair_pl']: batch_pcpair,
                     ops['flow_pl']: np.expand_dims(pc_pred-pc1,0),
                     ops['is_training_pl']: is_training}
        trans_pred, grouping_pred, seg_pred, conf_pred = sess.run([
            ops['pred_trans'], ops['pred_grouping'], ops['pred_seg'], ops['pred_conf']], feed_dict=feed_dict)

        ## final seg
        seg_gt = batch_seg[0].reshape(-1)
        seg_pred = np.argmax(seg_pred[0][np.squeeze(conf_pred[0])>0.5,:],0).reshape(-1)
        seg_pred = seg_util.seg_merge(pc1, pc_pred, seg_pred)
        RI_all.append(rand_index_score(seg_pred, seg_gt))
        EPE_all.append( np.mean(np.sqrt(np.sum((pc_pred-pc1_ini-batch_flow[0])**2,1))) )
    print('Mean RI: %f'%np.mean(np.array(RI_all)))
    print('Mean EPE: %f'%np.mean(np.array(EPE_all)))
    return 0

if __name__ == "__main__":
    evaluation()
