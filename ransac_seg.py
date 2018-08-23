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

def fit_motion_ransac(pc1, pc2, dist_th):
    nsmp = 1000
    npoint = pc1.shape[0]
    smpidx = np.random.randint(0, npoint, (nsmp,3))
    inlier_count = np.zeros(nsmp)
    for i in range(nsmp):
        R, t = fit_motion(pc1[smpidx[i,:],:], pc2[smpidx[i,:],:])
        inlier_count[i] = np.sum((np.sqrt(np.sum((np.matmul(pc1, R)+t-pc2)**2,1))<dist_th).astype('float32'))
    imx = np.argmax(inlier_count)
    R, t = fit_motion(pc1[smpidx[imx,:],:], pc2[smpidx[imx,:],:])
    inlier_idx = np.sqrt(np.sum((np.matmul(pc1, R)+t-pc2)**2,1))<dist_th
    R, t = fit_motion(pc1[inlier_idx,:], pc2[inlier_idx,:])
    return R, t

def ransac_seg_one_mode(pc1, pcPred, pc2, params):
    nsmp = params['nsmp_factor']*pc1.shape[0]
    ntrial = params['ntrial']
    smp_dist_disparity_th = params['smp_dist_disparity_th']
    corrs_dist_th = params['corrs_dist_th']
    inlier_eps = params['inlier_eps']
    maskk = np.min([params['maskk'],pcPred.shape[0]])
    nrefine_iter = params['nrefine_iter']
    issubsmp = params['issubsmp']
    nsubsmp1 = np.min([params['nsubsmp'],pc1.shape[0]])
    nsubsmp2 = np.min([params['nsubsmp'],pc2.shape[0]])
    if issubsmp:
        pc1_ini = copy.deepcopy(pc1)
        pc2_ini = copy.deepcopy(pc2)
        pcPred_ini = copy.deepcopy(pcPred)
        sidx1 = np.random.permutation(pc1_ini.shape[0])[:nsubsmp1]
        sidx2 = np.random.permutation(pc2_ini.shape[0])[:nsubsmp2]
        pc1 = pc1_ini[sidx1,:]
        pcPred = pcPred_ini[sidx1,:]
        pc2 = pc2_ini[sidx2,:]
    npt1 = pc1.shape[0]
    npt2 = pc2.shape[0]
    nbrs = NearestNeighbors(n_neighbors=maskk, algorithm='ball_tree').fit(pcPred)
    maskdist, maskidx = nbrs.kneighbors(pc2) # npt2 x maskk
    maskdist = maskdist<corrs_dist_th
    tmpmaskdist = np.reshape(np.transpose(maskdist),maskk*npt2)
    tmpmaskidx = np.reshape(np.transpose(maskidx),maskk*npt2)
    sidx = np.random.randint(0, npt1, (nsmp, 3))
    incount_best = 0
    R_best = np.eye(3)
    t_best = np.zeros((1,3))
    for j in range(nsmp):
        for k in range(ntrial):
            sidx[j,1] = np.random.randint(npt1)
            n1 = np.linalg.norm(pc1[sidx[j,0],:]-pc1[sidx[j,1],:])
            n2 = np.linalg.norm(pcPred[sidx[j,0],:]-pcPred[sidx[j,1],:])
            diff = n1-n2
            if np.abs(diff)<smp_dist_disparity_th and n1>0:
                break
        for k in range(ntrial):
            sidx[j,2] = np.random.randint(npt1)
            n11 = np.linalg.norm(pc1[sidx[j,0],:]-pc1[sidx[j,2],:])
            n12 = np.linalg.norm(pcPred[sidx[j,0],:]-pcPred[sidx[j,2],:])
            diff1 = n11-n12
            n21 = np.linalg.norm(pc1[sidx[j,1],:]-pc1[sidx[j,2],:])
            n22 = np.linalg.norm(pcPred[sidx[j,1],:]-pcPred[sidx[j,2],:])
            diff2 = n21-n22
            if np.abs(diff1)<smp_dist_disparity_th and np.abs(diff2)<smp_dist_disparity_th and n11>0 and n21>0:
                break
        R, t = fit_motion(pc1[sidx[j,:],:], pcPred[sidx[j,:],:])
        curtrans = np.matmul(pc1, R)+t
        tmpflag = tmpmaskdist * (np.sum(np.square(curtrans[tmpmaskidx,:]-np.tile(pc2,(maskk,1))),1)<inlier_eps*inlier_eps)
        tmpflag = np.reshape(tmpflag, (maskk, npt2))
        tmpflag = np.max(tmpflag,0)
        if np.sum(tmpflag)>incount_best:
            incount_best = np.sum(tmpflag)
            R_best = R
            t_best = t
    if issubsmp:
        pc1 = pc1_ini
        pc2 = pc2_ini
        pcPred = pcPred_ini
        npt1 = pc1.shape[0]
        npt2 = pc2.shape[0]
        nbrs = NearestNeighbors(n_neighbors=maskk, algorithm='ball_tree').fit(pcPred)
        maskdist, maskidx = nbrs.kneighbors(pc2) # npt2 x maskk
        maskdist = maskdist<corrs_dist_th
    ## refine mode
    R = R_best
    t = t_best
    for j in range(nrefine_iter):
        curtrans = np.matmul(pc1, R)+t
        tmpflag = np.zeros(npt2)
        for k in range(maskk):
            tmpflag = np.logical_or(tmpflag, maskdist[:,k] * (np.sqrt(np.sum((curtrans[maskidx[:,k],:]-pc2)**2,1))<inlier_eps) )
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

def ransac_seg_multi_modes(pc1, pcPred, pc2, params):
    nsegmax = params['nsegmax']
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pc1)
    dd, idx = nbrs.kneighbors(pc1)
    dist_th = np.max(dd[:,1])
    params['smp_dist_disparity_th'] = dist_th
    params['corrs_dist_th'] = dist_th
    params['inlier_eps'] = dist_th
    Rmodes = np.tile(np.expand_dims(np.eye(3),0),(nsegmax,1,1))
    tmodes = np.zeros((nsegmax,3))
    curpc1 = copy.deepcopy(pc1)
    curpcPred = copy.deepcopy(pcPred)
    curpc2 = copy.deepcopy(pc2)
    nmodes = 0
    segidx = np.zeros(curpc1.shape[0])
    segidx2 = np.zeros(curpc2.shape[0])
    ridx = np.arange(curpc1.shape[0])
    ridx2 = np.arange(curpc2.shape[0])
    for i in range(nsegmax):
        R, t, inlier_idx1, inlier_idx2 = ransac_seg_one_mode(curpc1, curpcPred, curpc2, params)
        Rmodes[i,...] = R
        tmodes[i,:] = t.reshape(3)
        curpc1 = curpc1[~inlier_idx1,:]
        segidx[ridx[inlier_idx1]] = i+1
        ridx = ridx[np.where(~inlier_idx1)[0]]
        curpcPred = curpcPred[~inlier_idx1,:]
        curpc2 = curpc2[~inlier_idx2,:]
        segidx2[ridx2[inlier_idx2]] = i+1
        ridx2 = ridx2[np.where(~inlier_idx2)[0]]
        nmodes += 1
        if curpc1.shape[0]<5 or curpc2.shape[0]<5:
            break
    return Rmodes, tmodes, nmodes, segidx, segidx2

def gen_attention_mask(pc1, segidx, nmodes, npoint):
    # segidx - 0 indicates unknown
    distmatrix = np.sqrt(np.sum((np.expand_dims(pc1, 1)-np.expand_dims(pc1,0))**2,2))
    if np.sum(segidx==0)>0:
        dist0 = np.zeros((np.sum(segidx==0),nmodes))
        for j in range(nmodes):
            if np.sum(segidx==j+1)>0:
                dist0[:,j] = np.min(distmatrix[segidx==0,:][:,segidx==j+1],1)
            else:
                dist0[:,j] = np.inf
        segidx[segidx==0] = np.argmin(dist0,1)+1
    pcout = np.zeros((nmodes, npoint, 3))
    attmask = np.zeros((nmodes, npoint), dtype='bool')
    for i in range(nmodes):
        if np.sum(segidx==i+1)>0:
            subidx1 = np.min(distmatrix[segidx==i+1,:],0)<0.1
            subpc1 = pc1[subidx1,:]
            smppc = pc1[segidx==i+1,:]
            pcout[i,:,:] = np.concatenate((subpc1, smppc[np.random.choice(np.arange(smppc.shape[0]),npoint-subpc1.shape[0]),:]),0)
            attmask[i,:] = subidx1
    return pcout, attmask, segidx

def gen_attention_mask2(pc1, pcPred, pc2, segidx, segidx2, Rmodes, tmodes, nmodes, npoint, distmatrix=None, distmatrix2=None, expand_eps=0.15):
    # segidx - 0 indicates unknown
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

    # batch_pc1 = np.zeros((nmodes, npoint, 3))
    # for i in range(nmodes):
    #     batch_pc1[i,:,:] = np.matmul(pc1, Rmodes[i,:,:])+tmodes[[i],:]
    # segidx_aug = np.argmin(np.sum((batch_pc1-np.expand_dims(pcPred,0))**2,2),0)+1
    # segidx2_aug = segidx_aug[np.argmin(np.sum((np.expand_dims(pc2,1)-np.expand_dims(pcPred,0))**2,2),1)]

    # segidx_aug = copy.deepcopy(segidx)
    # segidx2_aug = copy.deepcopy(segidx2)
    # if np.sum(segidx_aug==0)>0:
    #     dist0 = np.zeros((np.sum(segidx_aug==0),nmodes))
    #     for j in range(nmodes):
    #         if np.sum(segidx_aug==j+1)>0:
    #             dist0[:,j] = np.min(distmatrix[segidx_aug==0,:][:,segidx_aug==j+1],1)
    #         else:
    #             dist0[:,j] = np.inf
    #     segidx_aug[segidx_aug==0] = np.argmin(dist0,1)+1
    # if np.sum(segidx2_aug==0)>0:
    #     dist0 = np.zeros((np.sum(segidx2_aug==0),nmodes))
    #     for j in range(nmodes):
    #         if np.sum(segidx2_aug==j+1)>0:
    #             dist0[:,j] = np.min(distmatrix2[segidx2_aug==0,:][:,segidx2_aug==j+1],1)
    #         else:
    #             dist0[:,j] = np.inf
    #     segidx2_aug[segidx2_aug==0] = np.argmin(dist0,1)+1

    batch_pc1 = np.zeros((nmodes, npoint, 3))
    for i in range(nmodes):
        batch_pc1[i,:,:] = np.matmul(pc1, Rmodes[i,:,:])+tmodes[[i],:]
    spatial_dist_weights = np.ones((nmodes, npoint))
    for j in range(nmodes):
        if np.sum(segidx==j+1)>0:
            spatial_dist_weights[j,:] = np.min(distmatrix[:,segidx==j+1],1)
    segidx_aug = np.argmin(np.sum((batch_pc1-np.expand_dims(pcPred,0))**2,2)*spatial_dist_weights,0)+1
    segidx2_aug = segidx_aug[np.argmin(np.sum((np.expand_dims(pc2,1)-np.expand_dims(pcPred,0))**2,2),1)]


    pcout1 = np.zeros((nmodes, npoint, 3))
    pcout2 = np.zeros((nmodes, npoint, 3))
    attmask1 = np.zeros((nmodes, npoint), dtype='bool')
    attmask2 = np.zeros((nmodes, npoint), dtype='bool')
    for i in range(nmodes):
        if np.sum(segidx==i+1)>0 or np.sum(segidx_aug==i+1)>0:
            subidx1 = np.min(distmatrix[np.logical_or(segidx==i+1,segidx_aug==i+1),:],0)<expand_eps
            subpc1 = pc1[subidx1,:]
            smppc = pc1[np.logical_or(segidx==i+1,segidx_aug==i+1),:]
            pcout1[i,:,:] = np.concatenate((subpc1, smppc[np.random.choice(np.arange(smppc.shape[0]),npoint-subpc1.shape[0]),:]),0)
            attmask1[i,:] = subidx1
        if np.sum(segidx2==i+1)>0 or np.sum(segidx2_aug==i+1)>0:
            subidx2 = np.min(distmatrix2[np.logical_or(segidx2==i+1,segidx2_aug==i+1),:],0)<expand_eps
            subpc2 = pc2[subidx2,:]
            smppc2 = pc2[np.logical_or(segidx2==i+1,segidx2_aug==i+1),:]
            pcout2[i,:,:] = np.concatenate((subpc2, smppc2[np.random.choice(np.arange(smppc2.shape[0]),npoint-subpc2.shape[0]),:]),0)
            attmask2[i,:] = subidx2
    return pcout1, pcout2, attmask1, attmask2, segidx_aug, segidx2_aug, distmatrix, distmatrix2

def gen_attention_mask3(pc1, pcPred, pc2, segidx, segidx2, Rmodes, tmodes, nmodes, npoint, distmatrix=None, distmatrix2=None, expand_eps=0.15):
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
        # if np.sum(segidx2_aug==i+1)>0:
        #     subidx2 = np.min(distmatrix2[segidx2_aug==i+1,:],0)<expand_eps
        #     subpc2 = pc2[subidx2,:]
        #     smppc2 = pc2[segidx2_aug==i+1,:]
        #     pcout2[i,:,:] = np.concatenate((subpc2, smppc2[np.random.choice(np.arange(smppc2.shape[0]),npoint-subpc2.shape[0]),:]),0)
        #     attmask2[i,:] = subidx2
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
    mm = np.mean(np.min(curflownorm_aggreg,1))
    for i in range(nmode):
        modeindices = np.array(list(itertools.combinations(range(nmode), i+1)))
        scores = np.zeros(modeindices.shape[0])
        for j in range(modeindices.shape[0]):
            scores[j] = np.mean(np.min(curflownorm_aggreg[:,modeindices[j,:]],1))
        mmscore = np.min(scores)
        imx = np.argmin(scores)
        if mmscore<1.2*mm:
            modeidx = modeindices[imx,:]
            break

    imx = np.argmin(curflownorm_aggreg[:,modeidx],1)
    distmask = np.zeros(npt*nmode)
    distmask[np.ravel_multi_index((np.arange(npt),modeidx[imx]),(npt, nmode))] = 1.0
    distmask = np.reshape(distmask,(npt,nmode))

    mask = np.reshape(distmask,(npt,1,nmode))
    curpred_aggreg = curpred_aggreg*mask
    curpred_aggreg = np.divide(np.sum(curpred_aggreg,2),np.sum(mask,2)+1e-8)
    segidx = np.argmax(distmask[:,modeidx],1)
    return curpred_aggreg, segidx

def motion_modes_aggreg_watt2(pc1, pcPred, attmask):
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

def motion_modes_aggreg(pc1, pcPred, pc2, pcPred_prev, params):
    # pc1, pcPred, pc2: Nmodes x Npt x 3
    # pcPred_prev: Npt x 3
    # inlier_eps = params['inlier_eps']
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pc1[0,:,:])
    dd, idx = nbrs.kneighbors(pc1[0,:,:])
    dist_th = np.max(dd[:,1])
    inlier_eps = dist_th
    nmode = pc1.shape[0]
    npt = pc1.shape[1]
    curpred_aggreg = np.zeros((npt,3,nmode))
    curflownorm_aggreg = np.zeros((npt,nmode))
    curdist_aggreg = np.zeros((npt,nmode))
    curdistidx_aggreg = np.zeros((npt,nmode),dtype='int32')
    for i in range(nmode):
        curpc1 = pc1[i,...]
        curpc2 = pc2[i,...]
        curpred = pcPred[i,...]
        curflowpred = curpred-curpc1
        curflownorm_aggreg[:,i] = np.sqrt(np.sum(curflowpred**2,1))
        curpred_aggreg[:,:,i] = curpred
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(curpred)
        curdist, curdistidx = nbrs.kneighbors(curpc2) # npt2 x 1
        curdist_aggreg[:,i] = curdist[:,0]
        curdistidx_aggreg[:,i] = curdistidx[:,0]
    mm = np.mean(np.min(curflownorm_aggreg,1))
    for i in range(nmode):
        modeindices = np.array(list(itertools.combinations(range(nmode), i+1)))
        scores = np.zeros(modeindices.shape[0])
        for j in range(modeindices.shape[0]):
            scores[j] = np.mean(np.min(curflownorm_aggreg[:,modeindices[j,:]],1))
        mmscore = np.min(scores)
        imx = np.argmin(scores)
        if mmscore<1.2*mm:
            modeidx = modeindices[imx,:]
            break

    # imx = np.argmin(np.sum((pc1[modeidx,:,:]-np.expand_dims(pcPred_prev,0))**2,2),0)
    # distmask = np.zeros(npt*nmode)
    # distmask[np.ravel_multi_index((np.arange(npt),modeidx[imx]),(npt, nmode))] = 1.0
    # distmask = np.reshape(distmask,(npt,nmode))

    imx = np.argmin(curflownorm_aggreg[:,modeidx],1)
    distmask = np.zeros(npt*nmode)
    distmask[np.ravel_multi_index((np.arange(npt),modeidx[imx]),(npt, nmode))] = 1.0
    distmask = np.reshape(distmask,(npt,nmode))
    dd = np.sort(curdist_aggreg[:,modeidx],1)
    imx = np.argmin(curdist_aggreg[:,modeidx],1)
    if dd.shape[1]>1:
        imx[(dd[:,1]-dd[:,0])<=inlier_eps] = -1.0
    for i in range(len(modeidx)):
        distmask[curdistidx_aggreg[imx==i,modeidx[i]],:] = 0.0
    for i in range(len(modeidx)):
        distmask[curdistidx_aggreg[imx==i,modeidx[i]],modeidx[i]] = 1.0
    flag = np.where(np.sum(distmask,1)>=2)[0]
    for i in flag:
        eqmodes = np.where(distmask[i,:]>0)[0]
        imx = np.argmax(curflownorm_aggreg[i,eqmodes])
        distmask[i,:] = 0.0
        distmask[i,eqmodes[imx]] = 1.0

    mask = np.reshape(distmask,(npt,1,nmode))
    curpred_aggreg = curpred_aggreg*mask
    curpred_aggreg = np.divide(np.sum(curpred_aggreg,2),np.sum(mask,2)+1e-8)
    segidx = np.argmax(distmask[:,modeidx],1)
    return curpred_aggreg, segidx

def decode_motion_modes_affine(xyz, flow_pred, xyz2, trans_pred, grouping_pred, seg_pred, select_pred, conf_pred, eps):
    # flow_pred: npoint x 3
    # trans_pred: nsmp x 18
    # grouping_pred: npoint x npoint
    # seg_pred: nmask x npoint
    # select_pred: nmask x nsmp
    # conf_pred: nmask
    npoint = seg_pred.shape[1]
    npoint2 = xyz2.shape[0]
    pcPred = xyz+flow_pred
    nmodes = np.maximum(np.sum(conf_pred>0.5),1) #0.5 for syn data and 0.7 for real data
    # nmodes = 2
    Rmodes = np.tile(np.expand_dims(np.eye(3),0),(nmodes,1,1))
    tmodes = np.zeros((nmodes,3))
    vidxmodes = np.ones(nmodes)
    for i in range(nmodes):
        if np.sum(seg_pred[i,:]>0.5)<5:
            vidxmodes[i] = 0
        else:
            sidx = np.argmax(select_pred[i,:])
            Rtmp = np.reshape(trans_pred[sidx,:9],(3,3))
            Rmodes[i,:,:] = copy.deepcopy(Rtmp+np.eye(3))
            tmodes[i,:] = copy.deepcopy(-np.matmul(trans_pred[[sidx],15:],Rtmp)+trans_pred[[sidx],9:12]+trans_pred[[sidx],12:15])
    segidx = seg_pred[:nmodes,:]
    nmodes = np.sum(vidxmodes).astype('int32')
    Rmodes = Rmodes[vidxmodes==1,:,:]
    tmodes = tmodes[vidxmodes==1,:]
    segidx = segidx[vidxmodes==1,:]
    segidx = np.argmax(segidx,0) # (npoint)
    segidx2 = np.inf*np.ones((nmodes, npoint2))
    for i in range(nmodes):
        if np.sum(segidx==i)>0:
            segidx2[i,:] = np.min(np.sum((np.expand_dims(xyz2,1)-np.expand_dims(pcPred[segidx==i,:],0))**2,2),1)
    segidx2 = np.argmin(segidx2,0)
    segidx = np.eye(nmodes)[segidx,:]
    segidx2 = np.eye(nmodes)[segidx2,:]
    return Rmodes, tmodes, nmodes, segidx, segidx2

def decode_motion_modes(xyz, flow_pred, xyz2, trans_pred, grouping_pred, seg_pred, select_pred, conf_pred, eps):
    # flow_pred: npoint x 3
    # trans_pred: nsmp x 18
    # grouping_pred: npoint x npoint
    # seg_pred: nmask x npoint
    # select_pred: nmask x nsmp
    # conf_pred: nmask
    npoint = seg_pred.shape[1]
    nmodes = np.maximum(np.sum(conf_pred>0.5),1) #0.5 for syn data and 0.7 for real data
    # nmodes = 2
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
            R, t, inlier_idx1, inlier_idx2 = fit_motion_to_seg(xyz, xyz2, flow_pred, sidx, eps)
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

def fit_motion_to_seg(pc1, pc2, flow12, segidx, eps):
    # segidx: (npoint), segmentation index on pc1
    # pc1: (npoint, 3)
    # pc2: (npoint, 3), a partial point cloud with potential padding to form npoint
    # flow12: (npoint, 3) a flow field from pc1 to pc2
    th = np.maximum(np.sort(segidx)[20], 0.5)
    # th = np.minimum(np.sort(segidx)[15], 0.5)
    segidx = (segidx>th).astype('float32')
    maskk = 8
    npt1 = pc1.shape[0]
    npt2 = pc2.shape[0]
    pcpred = pc1+flow12
    nbrs = NearestNeighbors(n_neighbors=maskk, algorithm='ball_tree').fit(pcpred)
    maskdist, maskidx = nbrs.kneighbors(pc2) # npt2 x maskk
    # maskdist = maskdist<eps
    tmpmaskidx = np.reshape(np.transpose(maskidx),maskk*npt2)
    vmask = np.reshape(np.cumsum(np.reshape(segidx[tmpmaskidx],(maskk, npt2)),0)==1,maskk*npt2).astype('float32')
    # p2mask = np.tile(np.reshape(np.sum(np.reshape(segidx[tmpmaskidx],(maskk, npt2)),0)>=maskk/2,(1,npt2)),(maskk,1))
    p2mask = np.tile(np.reshape(np.sum(np.reshape(segidx[tmpmaskidx],(maskk, npt2)),0)>0,(1,npt2)),(maskk,1))
    p2mask = np.reshape(p2mask,maskk*npt2).astype('float32')
    vmask = vmask*p2mask
    tmppc1 = pc1[tmpmaskidx,:]
    tmppc2 = np.tile(pc2,(maskk,1))
    tmppc1 = tmppc1[np.logical_and(segidx[tmpmaskidx]>0,vmask),:]
    tmppc2 = tmppc2[np.logical_and(segidx[tmpmaskidx]>0,vmask),:]
    if tmppc1.shape[0]>5:
        # R, t = fit_motion_ransac(tmppc1, tmppc2, eps)
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
            # R, t = fit_motion_ransac(tmppc1, tmppc2, eps)
            R, t = fit_motion(tmppc1, tmppc2)
    inlier_idx2 = tmpflag
    inlier_idx1 = np.zeros(npt1)==1
    if tmppc1.shape[0]>0:
        nbrs1 = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pc1)
        _, pc1flag = nbrs1.kneighbors(tmppc1)
        pc1flag = np.unique(pc1flag)
        inlier_idx1[pc1flag] = True
    return R, t, inlier_idx1, inlier_idx2

def normalize_pcpair(pc1, pc2):
    pccat = np.concatenate((pc1,pc2),0)
    cc = (np.amax(pccat,0,keepdims=True)+np.amin(pccat,0,keepdims=True))/2.0
    pc1_post = pc1-cc
    pc2_post = pc2-cc
    s = np.sqrt(np.sum((np.amax(pc1,0)-np.amin(pc1,0))**2))+np.sqrt(np.sum((np.amax(pc2,0)-np.amin(pc2,0))**2))
    s = s/2.0
    # s = np.sqrt(np.sum((np.amax(pccat,0)-np.amin(pccat,0))**2))
    # s = 1.0
    if s<0.1:
        s = 1.0
    pc1_post = pc1_post/s
    pc2_post = pc2_post/s
    return pc1_post, pc2_post, s, cc

def denormalize_pc(pc1, s, cc):
    pc1_post = pc1*s
    pc1_post = pc1_post+cc
    return pc1_post

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

if __name__=='__main__':
    data = sio.loadmat('/Users/yl/Study/Projects/PartMobility/Code/matlab/gen_deformed_snc/tmp_fullshapepair2rt_smoothed_hu_data2.mat')
    pc1 = data['img'][9,:,:3]
    pc2 = data['img'][9,:,3:6]
    pcPred = pc1+data['flow_1corrs'][9,:,:]
    R, t = fit_motion(pc1, pcPred)
    pc1 = np.matmul(pc1,R)+t
    params = dict()
    params['nsmp'] = 15000
    params['smp_dist_disparity_th'] = 0.03
    params['corrs_dist_th'] = 0.03
    params['inlier_eps'] = 0.01
    params['maskk'] = 15
    params['nrefine_iter'] = 5
    params['issubsmp'] = True
    params['nsubsmp'] = 360
    params['nsegmax'] = 5
    start_time = time.time()
    Rmodes, tmodes, nmodes, segidx = ransac_seg_multi_modes(pc1, pcPred, pc2, params)
    # R, t, inlier_idx1, inlier_idx2 = ransac_seg_one_mode(pc1, pcPred, pc2, params)
    data = dict()
    data['Rmodes'] = Rmodes
    data['tmodes'] = tmodes
    data['nmodes'] = nmodes
    data['segidx'] = segidx
    # data['R'] = R
    # data['t'] = t
    # data['inlier_idx1'] = inlier_idx1
    # data['inlier_idx2'] = inlier_idx2
    sio.savemat('/Users/yl/Study/Projects/PartMobility/Code/matlab/gen_deformed_snc/tmp.mat',data)
    print(time.time()-start_time)