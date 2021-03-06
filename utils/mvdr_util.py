import librosa
import numpy as np


def make_wta(result_mask):
    noise_mask = result_mask[2]
    if len(result_mask) == 4:
        noise_mask += result_mask[3]
    mask = np.stack((result_mask[0], result_mask[1],noise_mask))
    mask_max = np.amax(mask, axis=0, keepdims=True)
    mask = np.where(mask==mask_max, mask, 1e-10)
    return mask


def make_mvdr(s,result):
    mask=make_wta(result)
    M=[]
    for i in range(7):
        st=librosa.core.stft(s[:,i],n_fft=512,hop_length=256)
        M.append(st)
    M=np.asarray(M)

    L=np.min([mask.shape[-1],M.shape[-1]])
    M=M[:,:,:L]

    mask=mask[:,:,:L]

    tgt_scm,_=get_mask_scm(M,mask[0])
    itf_scm,_=get_mask_scm(M,mask[1])
    noi_scm,_=get_mask_scm(M,mask[2])

    coef=calc_bfcoeffs(noi_scm+itf_scm,tgt_scm)
    res=get_bf(M,coef)
    res1=librosa.istft(res,hop_length=256)

    coef=calc_bfcoeffs(noi_scm+tgt_scm,itf_scm)
    res=get_bf(M,coef)
    res2=librosa.istft(res,hop_length=256)

    return res1, res2


def get_mask_scm(mix,mask):
    Ri = np.einsum('FT,FTM,FTm->FMm', mask, mix.transpose(1,2,0), mix.transpose(1,2,0).conj())
    t1=np.eye(7)
    t2=t1[np.newaxis,:,:]
    Ri+=1e-15*t2
    return Ri,np.sum(mask)


def calc_bfcoeffs(noi_scm,tgt_scm):
    # Calculate BF coeffs.
    num = np.linalg.solve(noi_scm, tgt_scm)
    den = np.trace(num, axis1=-2, axis2=-1)[..., np.newaxis, np.newaxis]
    den[0]+=1e-15
    W = (num / den)[..., 0]
    return W


def get_bf(mix,W):
    c,f,t=mix.shape
    return np.sum(W.reshape(f,c,1).conj()*mix.transpose(1,0,2),axis=1)
