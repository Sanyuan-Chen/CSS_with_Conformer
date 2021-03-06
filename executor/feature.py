#!/usr/bin/env python

"""
Implementation of front-end feature via PyTorch
"""

import math
import torch as th

from collections.abc import Sequence

import torch.nn.functional as F
import torch.nn as nn

EPSILON = th.finfo(th.float32).eps
MATH_PI = math.pi


def init_kernel(frame_len,
                frame_hop,
                normalize=True,
                round_pow_of_two=True,
                window="sqrt_hann"):
    if window != "sqrt_hann" and window != "hann":
        raise RuntimeError("Now only support sqrt hanning window or hann window")
    # FFT points
    N = 2**math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    # window
    W = th.hann_window(frame_len)
    if window == "sqrt_hann":
        W = W**0.5
    # scale factor to make same magnitude after iSTFT
    if window == "sqrt_hann" and normalize:
        S = 0.5 * (N * N / frame_hop)**0.5
    else:
        S = 1
    # F x N/2+1 x 2
    K = th.rfft(th.eye(N) / S, 1)[:frame_len]
    # 2 x N/2+1 x F
    K = th.transpose(K, 0, 2) * W
    # N+2 x 1 x F
    K = th.reshape(K, (N + 2, 1, frame_len))
    return K


class STFTBase(nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it 
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """
    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 normalize=True,
                 round_pow_of_two=True):
        super(STFTBase, self).__init__()
        K = init_kernel(frame_len,
                        frame_hop,
                        round_pow_of_two=round_pow_of_two,
                        window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window
        self.normalize = normalize
        self.num_bins = self.K.shape[0] // 2
        if window == "hann":
            self.conjugate = True
        else:
            self.conjugate = False

    def extra_repr(self):
        return (f"window={self.window}, stride={self.stride}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}, " +
                f"normalize={self.normalize}")


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x, cplx=False):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, N x C x S or N x S
        return
            m: magnitude, N x C x F x T or N x F x T
            p: phase, N x C x F x T or N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__name__, x.dim()))
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = th.unsqueeze(x, 1)
            # N x 2F x T
            c = F.conv1d(x, self.K, stride=self.stride, padding=0)
            # N x F x T
            r, i = th.chunk(c, 2, dim=1)
            if self.conjugate:
                # to match with science pipeline, we need to do conjugate
                i = -i
        # else reshape NC x 1 x S
        else:
            N, C, S = x.shape
            x = x.view(N * C, 1, S)
            # NC x 2F x T
            c = F.conv1d(x, self.K, stride=self.stride, padding=0)
            # N x C x 2F x T
            c = c.view(N, C, -1, c.shape[-1])
            # N x C x F x T
            r, i = th.chunk(c, 2, dim=2)
            if self.conjugate:
                # to match with science pipeline, we need to do conjugate
                i = -i
        if cplx:
            return r, i
        m = (r**2 + i**2)**0.5
        p = th.atan2(i, r)
        return m, p


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, cplx=False, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        args
            m, p: N x F x T
        return
            s: N x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = th.unsqueeze(p, 0)
            m = th.unsqueeze(m, 0)
        if cplx:
            # N x 2F x T
            c = th.cat([m, p], dim=1)
        else:
            r = m * th.cos(p)
            i = m * th.sin(p)
            # N x 2F x T
            c = th.cat([r, i], dim=1)
        # N x 2F x T
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        # N x S
        s = s.squeeze(1)
        if squeeze:
            s = th.squeeze(s)
        return s


class IPDFeature(nn.Module):
    """
    Compute inter-channel phase difference
    """
    def __init__(self,
                 ipd_index="1,0;2,0;3,0;4,0;5,0;6,0",
                 cos=True,
                 sin=False,
                 ipd_mean_normalize_version=2,
                 ipd_mean_normalize=True):
        super(IPDFeature, self).__init__()
        split_index = lambda sstr: [
            tuple(map(int, p.split(","))) for p in sstr.split(";")
        ]
        # ipd index
        pair = split_index(ipd_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.ipd_index = ipd_index
        self.cos = cos
        self.sin = sin
        self.ipd_mean_normalize=ipd_mean_normalize
        self.ipd_mean_normalize_version=ipd_mean_normalize_version
        self.num_pairs = len(pair) * 2 if cos and sin else len(pair)

    def extra_repr(self):
        return f"ipd_index={self.ipd_index}, cos={self.cos}, sin={self.sin}"

    def forward(self, p):
        """
        Accept multi-channel phase and output inter-channel phase difference
        args
            p: phase matrix, N x C x F x T
        return
            ipd: N x MF x T
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                "{} expect 3/4D tensor, but got {:d} instead".format(
                    self.__name__, p.dim()))
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        N, _, _, T = p.shape
        pha_dif = p[:, self.index_l] - p[:, self.index_r]
        if self.ipd_mean_normalize:
            yr = th.cos(pha_dif)
            yi = th.sin(pha_dif)
            yrm = yr.mean(-1, keepdim=True)
            yim = yi.mean(-1, keepdim=True)
            if self.ipd_mean_normalize_version == 1:
                pha_dif = th.atan2(yi - yim, yr - yrm)
            elif self.ipd_mean_normalize_version == 2:
                pha_dif_mean = th.atan2(yim, yrm)
                pha_dif -= pha_dif_mean
            elif self.ipd_mean_normalize_version == 3:
                pha_dif_mean = pha_dif.mean(-1, keepdim=True)
                pha_dif -= pha_dif_mean
            else:
                # we only support version 1, 2 and 3
                raise RuntimeError(
                    "{} expect ipd_mean_normalization version 1 or version 2, but got {:d} instead".format(
                        self.__name__, self.ipd_mean_normalize_version))

        if self.cos:
            # N x M x F x T
            ipd = th.cos(pha_dif)
            if self.sin:
                # N x M x 2F x T, along frequency axis
                ipd = th.cat([ipd, th.sin(pha_dif)], 2)
        else:
            # th.fmod behaves differently from np.mod for the input that is less than -math.pi
            # i believe it is a bug
            # so we need to ensure it is larger than -math.pi by adding an extra 6 * math.pi
            #ipd = th.fmod(pha_dif + math.pi, 2 * math.pi) - math.pi
            ipd = pha_dif
        # N x MF x T
        ipd = ipd.view(N, -1, T)
        # N x MF x T
        return ipd


class AngleFeature(nn.Module):
    """
    Compute angle/directional feature
        1) num_doas == 1: we known the DoA of the target speaker
        2) num_doas != 1: we do not have that prior, so we sampled #num_doas DoAs 
                          and compute on each directions    
    """
    def __init__(self,
                 geometric="princeton",
                 sr=16000,
                 velocity=340,
                 num_bins=257,
                 num_doas=1,
                 af_index="1,0;2,0;3,0;4,0;5,0;6,0"):
        super(AngleFeature, self).__init__()
        if geometric not in ["princeton"]:
            raise RuntimeError(
                "Unsupported array geometric: {}".format(geometric))
        self.geometric = geometric
        self.sr = sr
        self.num_bins = num_bins
        self.num_doas = num_doas
        self.velocity = velocity
        split_index = lambda sstr: [
            tuple(map(int, p.split(","))) for p in sstr.split(";")
        ]
        # ipd index
        pair = split_index(af_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.af_index = af_index
        omega = th.tensor(
            [math.pi * sr * f / (num_bins - 1) for f in range(num_bins)])
        # 1 x F
        self.omega = nn.Parameter(omega[None, :], requires_grad=False)

    def _oracle_phase_delay(self, doa):
        """
        Compute oracle phase delay given DoA
        args
            doa: N
        return
            phi: N x C x F or N x D x C x F
        """
        device = doa.device
        if self.num_doas != 1:
            # doa is a unused, fake parameter
            N = doa.shape[0]
            # N x D
            doa = th.linspace(0, MATH_PI * 2, self.num_doas + 1,
                              device=device)[:-1].repeat(N, 1)
        # for princeton
        # M = 7, R = 0.0425, treat M_0 as (0, 0)
        #      *3    *2
        #
        #   *4    *0    *1
        #
        #      *5    *6
        if self.geometric == "princeton":
            R = 0.0425
            zero = th.zeros_like(doa)
            # N x 7 or N x D x 7
            tau = R * th.stack([
                zero, -th.cos(doa), -th.cos(MATH_PI / 3 - doa),
                -th.cos(2 * MATH_PI / 3 - doa),
                th.cos(doa),
                th.cos(MATH_PI / 3 - doa),
                th.cos(2 * MATH_PI / 3 - doa)
            ],
                               dim=-1) / self.velocity
            # (Nx7x1) x (1xF) => Nx7xF or (NxDx7x1) x (1xF) => NxDx7xF
            phi = th.matmul(tau.unsqueeze(-1), -self.omega)
            return phi
        else:
            return None

    def extra_repr(self):
        return (
            f"geometric={self.geometric}, af_index={self.af_index}, " +
            f"sr={self.sr}, num_bins={self.num_bins}, velocity={self.velocity}, "
            + f"known_doa={self.num_doas == 1}")

    def _compute_af(self, ipd, doa):
        """
        Compute angle feature
        args
            ipd: N x C x F x T
            doa: DoA of the target speaker (if we known that), N 
                 or N x D (we do not known that, sampling D DoAs instead)
        return
            af: N x F x T or N x D x F x T
        """
        # N x C x F or N x D x C x F
        d = self._oracle_phase_delay(doa)
        d = d.unsqueeze(-1)
        if self.num_doas == 1:
            dif = d[:, self.index_l] - d[:, self.index_r]
            # N x C x F x T
            af = th.cos(ipd - dif)
            # on channel dimention (mean or sum)
            af = th.mean(af, dim=1)
        else:
            # N x D x C x F x 1
            dif = d[:, :, self.index_l] - d[:, :, self.index_r]
            # N x D x C x F x T
            af = th.cos(ipd.unsqueeze(1) - dif)
            # N x D x F x T
            af = th.mean(af, dim=2)
        return af

    def forward(self, p, doa):
        """
        Accept doa of the speaker & multi-channel phase, output angle feature
        args
            doa: DoA of target/each speaker, N or [N, ...]
            p: phase matrix, N x C x F x T
        return
            af: angle feature, N x F* x T or N x D x F x T (known_doa=False)
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                "{} expect 3/4D tensor, but got {:d} instead".format(
                    self.__name__, p.dim()))
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        ipd = p[:, self.index_l] - p[:, self.index_r]

        if isinstance(doa, Sequence):
            if self.num_doas != 1:
                raise RuntimeError("known_doa=False, no need to pass "
                                   "doa as a Sequence object")
            # [N x F x T or N x D x F x T, ...]
            af = [self._compute_af(ipd, spk_doa) for spk_doa in doa]
            # N x F x T => N x F* x T
            af = th.cat(af, 1)
        else:
            # N x F x T or N x D x F x T
            af = self._compute_af(ipd, doa)
        return af


class FeatureExtractor(nn.Module):
    """
    A PyTorch module to handle spectral & spatial features
    """
    def __init__(self,
                 frame_len=512,
                 frame_hop=256,
                 normalize=True,
                 round_pow_of_two=True,
                 num_spks=2,
                 log_spectrogram=True,
                 mvn_spectrogram=True,
                 ipd_mean_normalize=True,
                 ipd_mean_normalize_version=2,
                 window="sqrt_hann",
                 ext_af=0,
                 ipd_cos=True,
                 ipd_sin=False,
                 ipd_index="1,4;2,5;3,6",
                 ang_index="1,0;2,0;3,0;4,0;5,0;6,0"
                 ):
        super(FeatureExtractor, self).__init__()
        # forward STFT
        self.forward_stft = STFT(frame_len,
                                 frame_hop,
                                 normalize=normalize,
                                 window=window,
                                 round_pow_of_two=round_pow_of_two)
        self.inverse_stft = iSTFT(frame_len,
                                  frame_hop,
                                  normalize=normalize,
                                  round_pow_of_two=round_pow_of_two)
        self.has_spatial = False
        num_bins = self.forward_stft.num_bins
        self.feature_dim = num_bins
        self.num_bins = num_bins
        self.num_spks = num_spks
        # add extra angle feature
        self.ext_af = ext_af

        # IPD or not
        self.ipd_extractor = None
        if ipd_index:
            self.ipd_extractor = IPDFeature(ipd_index,
                                            cos=ipd_cos,
                                            sin=ipd_sin,
                                            ipd_mean_normalize_version=ipd_mean_normalize_version,
                                            ipd_mean_normalize=ipd_mean_normalize)
            self.feature_dim += self.ipd_extractor.num_pairs * num_bins
            self.has_spatial = True
        # AF or not
        self.ang_extractor = None
        if ang_index:
            self.ang_extractor = AngleFeature(
                num_bins=num_bins,
                num_doas=1,  # must known the DoA
                af_index=ang_index)
            self.feature_dim += num_bins * self.num_spks * (1 + self.ext_af)
            self.has_spatial = True
        # BN or not
        self.mvn_mag = mvn_spectrogram
        # apply log or not
        self.log_mag = log_spectrogram

    def _check_args(self, x, doa):
        if x.dim() == 2 and self.has_spatial:
            raise RuntimeError("Got 2D (single channel) input and can "
                               "not extract spatial features")
        if self.ang_extractor is None and doa:
            raise RuntimeError("DoA is given and AF extractor "
                               "is not initialized")
        if self.ang_extractor and doa is None:
            raise RuntimeError("AF extractor is initialized, but DoA is None")
        num_af = self.num_spks * (self.ext_af + 1)
        if isinstance(doa, Sequence) and len(doa) != num_af:
            raise RuntimeError("Number of DoA do not match the " +
                               f"speaker number: {len(doa):d} vs {num_af:d}")

    def stft(self, x, cplx=False):
        return self.forward_stft(x, cplx=cplx)

    def istft(self, m, p, cplx=False):
        return self.inverse_stft(m, p, cplx=cplx)

    def compute_spectra(self, x):
        """
        Compute spectra features
        args
            x: N x C x S (multi-channel) or N x S (single channel)
        return:
            mag & pha: N x F x T or N x C x F x T
            feature: N x * x T
        """
        # mag & pha: N x C x F x T or N x F x T
        mag, pha = self.forward_stft(x)
        # ch0: N x F x T
        if mag.dim() == 4:
            f = th.clamp(mag[:, 0], min=EPSILON)
        else:
            f = th.clamp(mag, min=EPSILON)
        # log
        if self.log_mag:
            f = th.log(f)
        # mvn
        if self.mvn_mag:
            # f = self.mvn_mag(f)
            f = (f - f.mean(-1, keepdim=True)) / (f.std(-1, keepdim=True) +
                                                  EPSILON)
        return mag, pha, f

    def compute_spatial(self, x, doa=None, pha=None):
        """
        Compute spatial features
        args
            x: N x C x S (multi-channel)
            pha: N x C x F x T
        return
            feature: N x * x T
        """
        if pha is None:
            self._check_args(x, doa)
            # mag & pha: N x C x F x T
            _, pha = self.forward_stft(x)
        else:
            if pha.dim() != 4:
                raise RuntimeError("Expect phase matrix a 4D tensor, " +
                                   f"got {pha.dim()} instead")
        feature = []
        if self.has_spatial:
            if self.ipd_extractor:
                # N x C x F x T => N x MF x T
                ipd = self.ipd_extractor(pha)
                feature.append(ipd)
            if self.ang_extractor:
                # N x C x F x T => N x F* x T
                ang = self.ang_extractor(pha, doa)
                feature.append(ang)
        else:
            raise RuntimeError("No spatial features are configured")
        # N x * x T
        feature = th.cat(feature, 1)
        return feature

    def forward(self, x, doa=None, ref_channel=0):
        """
        args
            x: N x C x S (multi-channel) or N x S (single channel)
            doa: N or [N, ...] (for each speaker)
        return:
            mag & pha: N x F x T (if ref_channel is not None), N x C x F x T
            feature: N x * x T
        """
        self._check_args(x, doa)
        # mag & pha: N x C x F x T or N x F x T
        mag, pha, f = self.compute_spectra(x)
        feature = [f]
        if self.has_spatial:
            spatial = self.compute_spatial(x, pha=pha, doa=doa)
            feature.append(spatial)
        # N x * x T
        feature = th.cat(feature, 1)
        if mag.dim() == 4 and ref_channel is not None:
            return mag[:, ref_channel], pha[:, ref_channel], feature
        else:
            return mag, pha, feature