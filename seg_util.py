import os
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.io as sio
import time
import copy
import itertools
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph

def fit_motion(pc1, pc2):
    pc1_centered = pc1-np.mean(pc1, 0, keepdims=True)
    pc2_centered = pc2-np.mean(pc2, 0, keepdims=True)
    R = np.matmul(np.transpose(pc1_centered), pc2_centered)
    u,s,v = np.linalg.svd(R, full_matrices=True)
    R = np.matmul(u,v)
    if np.sum(np.abs(np.real(np.linalg.eig(R)[0])-1)<1e-5)==0:
        u[:,2] = -u[:,2]
        R = np.matmul(u,v)
    t = np.mean(pc2-np.matmul(pc1, R), 0, keepdims=True)
    return R, t

def gen_attention_mask(pc1, pcPred, pc2, segidx, segidx2, Rmodes, tmodes, nmodes, npoint, distmatrix=None, distmatrix2=None, expand_eps=0.15):
    # segidx - each column corresponds to one segment
    if distmatrix is None:
        distmatrix = np.sqrt(np.sum((np.expand_dims(pc1, 1)-np.expand_dims(pc1,0))**2,2))
        distmatrix2 = np.sqrt(np.sum((np.expand_dims(pc2, 1)-np.expand_dims(pc2,0))**2,2))
        nnbr = 5
        nbrs = NearestNeighbors(n_neighbors=nnbr+1, algorithm='ball_tree').fit(pc1)
        _, idx = nbrs.kneighbors(pc1)
        distmask = np.zeros(npoint*npoint)
        for i in range(nnbr):
            distmask[np.ravel_multi_index((np.arange(npoint),idx[:,i+1]),(npoint, npoint))] = 1.0
        distmask = np.reshape(distmask,(npoint,npoint))
        distmatrix = distmatrix*distmask
        nbrs2 = NearestNeighbors(n_neighbors=nnbr+1, algorithm='ball_tree').fit(pc2)
        _, idx2 = nbrs2.kneighbors(pc2)
        distmask2 = np.zeros(npoint*npoint)
        for i in range(nnbr):
            distmask2[np.ravel_multi_index((np.arange(npoint),idx2[:,i+1]),(npoint, npoint))] = 1.0
        distmask2 = np.reshape(distmask2,(npoint,npoint))
        distmatrix2 = distmatrix2*distmask2
        distmatrix = sparse.csr_matrix(distmatrix)
        distmatrix = csgraph.floyd_warshall(distmatrix, directed=False)
        distmatrix2 = sparse.csr_matrix(distmatrix2)
        distmatrix2 = csgraph.floyd_warshall(distmatrix2, directed=False)

    batch_pc1 = np.zeros((nmodes, npoint, 3))
    for i in range(nmodes):
        batch_pc1[i,:,:] = np.matmul(pc1, Rmodes[i,:,:])+tmodes[[i],:]
    spatial_dist_weights = np.ones((nmodes, npoint))
    for j in range(nmodes):
        if np.sum(segidx[:,j])>0:
            spatial_dist_weights[j,:] = np.min(distmatrix[:,segidx[:,j]>0],1)
    segidx_aug = np.argmin(np.sum((batch_pc1-np.expand_dims(pcPred,0))**2,2)*spatial_dist_weights,0)+1
    segidx2_aug = segidx_aug[np.argmin(np.sum((np.expand_dims(pc2,1)-np.expand_dims(pcPred,0))**2,2),1)]

    pcout1 = np.zeros((nmodes, npoint, 3))
    pcout2 = np.zeros((nmodes, npoint, 3))
    attmask1 = np.zeros((nmodes, npoint), dtype='bool')
    attmask2 = np.zeros((nmodes, npoint), dtype='bool')
    for i in range(nmodes):
        if np.sum(segidx[:,i])>0 or np.sum(segidx_aug==i+1)>0:
            subidx1 = np.min(distmatrix[np.logical_or(segidx[:,i]>0,segidx_aug==i+1),:],0)<expand_eps
            subpc1 = pc1[subidx1,:]
            smppc = pc1[np.logical_or(segidx[:,i]>0,segidx_aug==i+1),:]
            pcout1[i,:,:] = np.concatenate((subpc1, smppc[np.random.choice(np.arange(smppc.shape[0]),npoint-subpc1.shape[0]),:]),0)
            attmask1[i,:] = subidx1
        if np.sum(segidx2[:,i])>0 or np.sum(segidx2_aug==i+1)>0:
            subidx2 = np.min(distmatrix2[np.logical_or(segidx2[:,i]>0,segidx2_aug==i+1),:],0)<expand_eps
            subpc2 = pc2[subidx2,:]
            smppc2 = pc2[np.logical_or(segidx2[:,i]>0,segidx2_aug==i+1),:]
            pcout2[i,:,:] = np.concatenate((subpc2, smppc2[np.random.choice(np.arange(smppc2.shape[0]),npoint-subpc2.shape[0]),:]),0)
            attmask2[i,:] = subidx2
    return pcout1, pcout2, attmask1, attmask2, segidx_aug, segidx2_aug, distmatrix, distmatrix2

def motion_modes_aggreg_watt(pc1, pcPred, attmask):
    # pc1, pcPred: Nmodes x Npt x 3
    # attmask: Nmodes x Npt
    nmode = pc1.shape[0]
    npt = pc1.shape[1]
    curpred_aggreg = np.zeros((npt,3,nmode))
    curflownorm_aggreg = np.inf*np.ones((npt,nmode))
    for i in range(nmode):
        nsubpt = np.sum(attmask[i,:])
        curpred_aggreg[attmask[i,:],:,i] = pcPred[i,:nsubpt,:]
        curflowpred = pcPred[i,:nsubpt,:]-pc1[i,:nsubpt,:]
        curflownorm_aggreg[attmask[i,:],i] = np.sqrt(np.sum(curflowpred**2,1))
    modeidx = np.arange(nmode)

    imx = np.argmin(curflownorm_aggreg[:,modeidx],1)
    distmask = np.zeros(npt*nmode)
    distmask[np.ravel_multi_index((np.arange(npt),modeidx[imx]),(npt, nmode))] = 1.0
    distmask = np.reshape(distmask,(npt,nmode))

    mask = np.reshape(distmask,(npt,1,nmode))
    curpred_aggreg = curpred_aggreg*mask
    curpred_aggreg = np.divide(np.sum(curpred_aggreg,2),np.sum(mask,2)+1e-8)
    segidx = np.argmax(distmask[:,modeidx],1)
    return curpred_aggreg, segidx

def decode_motion_modes(xyz, flow_pred, xyz2, trans_pred, grouping_pred, seg_pred, conf_pred, eps):
    # flow_pred: npoint x 3
    # trans_pred: nsmp x 18
    # grouping_pred: npoint x npoint
    # seg_pred: nmask x npoint
    # conf_pred: nmask
    npoint = seg_pred.shape[1]
    nmodes = np.maximum(np.sum(conf_pred>0.5),1)
    Rmodes = np.tile(np.expand_dims(np.eye(3),0),(nmodes,1,1))
    tmodes = np.zeros((nmodes,3))
    vidxmodes = np.ones(nmodes)
    segidx = np.zeros((nmodes, npoint))
    segidx2 = np.zeros(segidx.shape)
    for i in range(nmodes):
        if np.sum(seg_pred[i,:]>0.5)<3:
            vidxmodes[i] = 0
        else:
            sidx = seg_pred[i,:]
            R, t, inlier_idx1, inlier_idx2 = fit_motion_per_seg(xyz, xyz2, flow_pred, sidx, eps)
            segidx[i,:] = inlier_idx1
            segidx2[i,:] = inlier_idx2
            Rmodes[i,:,:] = copy.deepcopy(R)
            tmodes[i,:] = copy.deepcopy(t)
            if np.sum(segidx)<5:
                vidxmodes[i] = 0
    nmodes = np.sum(vidxmodes).astype('int32')
    Rmodes = Rmodes[vidxmodes==1,:,:]
    tmodes = tmodes[vidxmodes==1,:]
    segidx = segidx[vidxmodes==1,:]
    segidx2 = segidx2[vidxmodes==1,:]
    segidx = np.transpose(segidx,(1,0))
    segidx2 = np.transpose(segidx2,(1,0))
    return Rmodes, tmodes, nmodes, segidx, segidx2

def fit_motion_per_seg(pc1, pc2, flow12, segidx, eps):
    # segidx: (npoint), segmentation index on pc1
    # pc1: (npoint, 3)
    # pc2: (npoint, 3), a partial point cloud with potential padding to form npoint
    # flow12: (npoint, 3) a flow field from pc1 to pc2
    th = np.maximum(np.sort(segidx)[20], 0.5)
    segidx = (segidx>th).astype('float32')
    maskk = 8
    npt1 = pc1.shape[0]
    npt2 = pc2.shape[0]
    pcpred = pc1+flow12
    nbrs = NearestNeighbors(n_neighbors=maskk, algorithm='ball_tree').fit(pcpred)
    maskdist, maskidx = nbrs.kneighbors(pc2) # npt2 x maskk
    tmpmaskidx = np.reshape(np.transpose(maskidx),maskk*npt2)
    vmask = np.reshape(np.cumsum(np.reshape(segidx[tmpmaskidx],(maskk, npt2)),0)==1,maskk*npt2).astype('float32')
    p2mask = np.tile(np.reshape(np.sum(np.reshape(segidx[tmpmaskidx],(maskk, npt2)),0)>0,(1,npt2)),(maskk,1))
    p2mask = np.reshape(p2mask,maskk*npt2).astype('float32')
    vmask = vmask*p2mask
    tmppc1 = pc1[tmpmaskidx,:]
    tmppc2 = np.tile(pc2,(maskk,1))
    tmppc1 = tmppc1[np.logical_and(segidx[tmpmaskidx]>0,vmask),:]
    tmppc2 = tmppc2[np.logical_and(segidx[tmpmaskidx]>0,vmask),:]
    if tmppc1.shape[0]>5:
        R, t = fit_motion(tmppc1, tmppc2)
    else:
        R = np.eye(3)
        t = np.zeros(3)
    maskdist = np.transpose(np.reshape(segidx[tmpmaskidx]>0.5,(maskk, npt2)),(1,0))
    nrefine_iter = 3
    for j in range(nrefine_iter):
        curtrans = np.matmul(pc1, R)+t
        tmpflag = np.zeros(npt2)
        for k in range(maskk):
            tmpflag = np.logical_or(tmpflag, maskdist[:,k] * (np.sqrt(np.sum((curtrans[maskidx[:,k],:]-pc2)**2,1))<eps) )
        tmpmaskidx = maskidx[tmpflag,:]
        tmpmaskdist = maskdist[tmpflag,:]
        tmppc2 = pc2[tmpflag,:]
        tmppc1 = pc1[tmpmaskidx[:,0],:]
        tmpdist = np.inf*np.ones(tmppc2.shape[0])
        for k in range(maskk):
            dd = np.sqrt(np.sum((curtrans[tmpmaskidx[:,k],:]-tmppc2)**2,1))
            flag = dd<tmpdist
            flag = np.logical_and(flag, tmpmaskdist[:,k])
            tmppc1[flag,:] = pc1[tmpmaskidx[flag,k],:]
            tmpdist[flag] = dd[flag]
        if tmppc1.shape[0]<3 or tmppc2.shape[0]<3:
            R = np.eye(3)
            t = np.zeros((1,3))
            inlier_idx1 = np.zeros(npt1)==1
            inlier_idx2 = np.zeros(npt2)==1
            return R, t, inlier_idx1, inlier_idx2
        else:
            R, t = fit_motion(tmppc1, tmppc2)
    inlier_idx2 = tmpflag
    inlier_idx1 = np.zeros(npt1)==1
    if tmppc1.shape[0]>0:
        nbrs1 = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pc1)
        _, pc1flag = nbrs1.kneighbors(tmppc1)
        pc1flag = np.unique(pc1flag)
        inlier_idx1[pc1flag] = True
    return R, t, inlier_idx1, inlier_idx2

def seg_merge(pc1, pcpred, segidx):
    # pc1: N x 3
    # pcpred: N x 3
    # segidx: N
    useg = np.unique(segidx)
    nmode = len(useg)
    npoint = pc1.shape[0]
    if nmode==1:
        return segidx
    Rmodes = list()
    tmodes = list()
    for i in range(nmode):
        R, t = fit_motion(pc1[segidx==useg[i],:], pcpred[segidx==useg[i],:])
        Rmodes.append(R)
        tmodes.append(t)
    dres = np.zeros((npoint, nmode))
    for i in range(nmode):
        dres[:,i] = np.sqrt(np.sum((np.matmul(pc1, Rmodes[i])+tmodes[i]-pcpred)**2,1))
    dmode = np.zeros((nmode, nmode))
    for i in range(nmode):
        dmode[i,:] = np.mean(dres[segidx==useg[i],:],0)
    mindres = np.mean(np.diag(dmode))
    dresth = np.minimum(0.06, mindres*3)
    for i in range(2, nmode+1):
        select_pool = list(itertools.combinations(np.arange(nmode), 2))
        scores = np.zeros(len(select_pool))
        for j in range(len(select_pool)):
            scores[j] = np.mean(np.min(dmode[select_pool[j],:],0))
        if np.min(scores)<dresth:
            break
    imx = np.argmin(scores,0)
    imx = np.argmin(dmode[select_pool[imx],:],0)
    segidx_merged = np.zeros_like(segidx)
    for i in range(nmode):
        segidx_merged[segidx==useg[i]]=imx[i]
    if np.max(segidx_merged)==0 and nmode==2:
        segidx_merged = segidx
    return segidx_merged

def fps(pc1, Nout):
    Nin = pc1.shape[0]
    if Nout>Nin:
        pcout1 = np.concatenate((pc1, pc1[np.random.choice(np.arange(Nin),Nout-Nin),:]),0)
    else:
        selectIdx = np.zeros(Nin)
        seed = np.random.randint(Nin)
        selectIdx[seed] = 1
        count = 1
        pd = np.sum((pc1-pc1[[seed],:])**2,1)
        while count<Nout:
            imx = np.argmax(pd)
            selectIdx[imx] = 1
            count += 1
            pd = np.minimum(pd, np.sum((pc1-pc1[[imx],:])**2,1))
        pcout1 = pc1[selectIdx==1,:]
    return pcout1