import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import scipy.io as sio
import data_prep

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--stage', type=int, default=1, help='Training stage [default: 1]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.0005]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
STAGE = FLAGS.stage
NPOINT = 512
NMASK = 10

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(MODEL_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir+str(STAGE)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

#### Get data ####
if STAGE==1:
    TRAIN_DATASET = data_prep.FlowDataset('data/flow_train.mat', npoint=NPOINT)
    VAD_DATASET = data_prep.FlowDataset('data/flow_validation.mat', npoint=NPOINT)
else:
    if STAGE==2:
        RELROT = True
    else:
        RELROT = False
    TRAIN_DATASET = data_prep.SegDataset('data/seg_train.mat', npoint=NPOINT, relrot=RELROT)
    VAD_DATASET = data_prep.SegDataset('data/seg_validation.mat', npoint=NPOINT, relrot=RELROT)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pcpair_pl, flow_pl, vismask_pl, momasks_pl = MODEL.placeholder_inputs(NMASK, NPOINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            if STAGE==1:
                pred_flow, pred_vismask, loss1, loss2, loss3, loss = MODEL.get_model_loss_stage1(pcpair_pl, vismask_pl, flow_pl, momasks_pl, is_training_pl, bn_decay)
            elif STAGE==2:
                pred_trans, pred_grouping_sub, loss1, loss2, loss = MODEL.get_model_loss_stage2(pcpair_pl, vismask_pl, flow_pl, momasks_pl, is_training_pl, bn_decay)
                loss3 = 0*loss
            else:
                pred_seg_sub, pred_conf, loss, loss1, loss2 = MODEL.get_model_loss_stage3(pcpair_pl, vismask_pl, flow_pl, momasks_pl, is_training_pl, bn_decay)
                loss3 = 0*loss

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('loss1', loss1)
            tf.summary.scalar('loss2', loss2)
            tf.summary.scalar('loss3', loss3)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            if STAGE>2:
                saver_corrsflow = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CorrsFeaExtractor')+
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CorrsFlowNet'))
                saver_transgrouping = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='TransNet')+
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='GroupingNet'))
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        if STAGE>2:
            saver_corrsflow.restore(sess, FLAGS.log_dir+'1/best_model.ckpt')
            saver_transgrouping.restore(sess, FLAGS.log_dir+'2/best_model.ckpt')

        ops = {'pcpair_pl': pcpair_pl,
               'flow_pl': flow_pl,
               'vismask_pl': vismask_pl,
               'momasks_pl': momasks_pl,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'loss1': loss1,
               'loss2': loss2,
               'loss3': loss3,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        best_loss = np.inf
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            train_one_epoch(sess, ops, train_writer)
            loss = eval_one_epoch(sess, ops, test_writer)
            if loss < best_loss:
                best_loss = loss
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def get_batch_data(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_pcpair = np.zeros((bsize, NPOINT, 6), dtype=np.float)
    batch_flow = np.zeros((bsize, NPOINT, 3), dtype=np.float)
    batch_vismask = np.zeros((bsize, NPOINT), dtype=np.float)
    batch_momasks = np.zeros((bsize, NMASK, NPOINT), dtype=np.float)
    for i in range(bsize):
        pc1, pc2, flow12, vismask, momasks = dataset[idxs[i+start_idx]]
        batch_pcpair[i,...] = np.concatenate((pc1,pc2), 1)
        batch_flow[i,...] = flow12
        batch_vismask[i,:] = vismask
        batch_momasks[i,...] = np.transpose(momasks)
    return batch_pcpair, batch_flow, batch_vismask, batch_momasks

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = np.minimum(np.int(len(TRAIN_DATASET)/BATCH_SIZE),10000)
    loss_sum = 0
    loss1_sum = 0
    loss2_sum = 0
    loss3_sum = 0
    
    log_string(str(datetime.now()))
    for batch_idx in range(num_batches):
        # Get batch data
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_pcpair, batch_flow, batch_vismask, batch_momasks = get_batch_data(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        # Feed data
        feed_dict = {ops['pcpair_pl']: batch_pcpair,
                     ops['flow_pl']: batch_flow,
                     ops['vismask_pl']: batch_vismask,
                     ops['momasks_pl']: batch_momasks,
                     ops['is_training_pl']: is_training}
        summary, step, _, loss_val, loss1_val, loss2_val, loss3_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['loss1'], ops['loss2'], ops['loss3']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        loss_sum += loss_val
        loss1_sum += loss1_val
        loss2_sum += loss2_val
        loss3_sum += loss3_val

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            if STAGE==1:
                log_string('mean loss: %f, mean loss_flow: %f, mean loss_vismask: %f, mean loss_matching: %f' % (loss_sum / 10, loss1_sum / 10, loss2_sum / 10, loss3_sum / 10))
            elif STAGE==2:
                log_string('mean loss: %f, mean loss_trans: %f, mean loss_grouping: %f' % (loss_sum / 10, loss1_sum / 10, loss2_sum / 10))
            else:
                log_string('mean loss: %f, mean negative iou: %f, mean loss_seg: %f' % (loss_sum / 10, loss2_sum / 10, loss1_sum / 10))
            loss_sum = 0
            loss1_sum = 0
            loss2_sum = 0
            loss3_sum = 0

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(VAD_DATASET))
    np.random.shuffle(test_idxs)
    num_batches = np.int(len(VAD_DATASET)/BATCH_SIZE)
    loss_sum = 0
    loss1_sum = 0
    loss2_sum = 0
    loss3_sum = 0
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_pcpair, batch_flow, batch_vismask, batch_momasks = get_batch_data(VAD_DATASET, test_idxs, start_idx, end_idx)
        feed_dict = {ops['pcpair_pl']: batch_pcpair,
                     ops['flow_pl']: batch_flow,
                     ops['vismask_pl']: batch_vismask,
                     ops['momasks_pl']: batch_momasks,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, loss1_val, loss2_val, loss3_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['loss1'], ops['loss2'], ops['loss3']], feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        loss_sum += loss_val
        loss1_sum += loss1_val
        loss2_sum += loss2_val
        loss3_sum += loss3_val

    EPOCH_CNT += 1

    if STAGE==1:
        log_string('eval mean loss: %f, eval mean loss_flow: %f, eval mean loss_vismask: %f, eval mean loss_matching: %f' % (loss_sum / float(num_batches), loss1_sum / float(num_batches), loss2_sum / float(num_batches), loss3_sum / float(num_batches))) 
        return loss1_sum / float(num_batches)
    elif STAGE==2:
        log_string('eval mean loss: %f, eval mean loss_trans: %f, eval mean loss_grouping: %f' % (loss_sum / float(num_batches), loss1_sum / float(num_batches), loss2_sum / float(num_batches))) 
        return loss_sum / float(num_batches)
    else:
        log_string('eval mean loss: %f, eval mean negative iou: %f, eval mean loss_seg: %f' % (loss_sum / float(num_batches), loss2_sum / float(num_batches), loss1_sum / float(num_batches))) 
        return loss_sum / float(num_batches)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
