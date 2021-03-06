import os
import numpy as np
import soundfile as sf
import scipy.io.wavfile as wf

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps


def _parse_script(scp_path,
                  value_processor=lambda x: x,
                  num_tokens=2,
                  restrict=True):
    """
    Parse kaldi's script(.scp) file
    If num_tokens >= 2, function will check token number
    """
    scp_dict = dict()
    line = 0
    with open(scp_path, "r") as f:
        for raw_line in f:
            scp_tokens = raw_line.strip().split()
            line += 1
            if (num_tokens >= 2 and len(scp_tokens) != num_tokens) or (
                    restrict and len(scp_tokens) < 2):
                raise RuntimeError(
                    "For {}, format error in line[{:d}]: {}".format(
                        scp_path, line, raw_line))
            if num_tokens == 2:
                key, value = scp_tokens
            else:
                key, value = scp_tokens[0], scp_tokens[1:]
            if key in scp_dict:
                raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                    key, scp_path))
            scp_dict[key] = value_processor(value)
    return scp_dict


class BaseReader(object):
    """
        BaseReader Class
    """
    def __init__(self, scp_rspecifier, **kwargs):
        self.index_dict = _parse_script(scp_rspecifier, **kwargs)
        self.index_keys = list(self.index_dict.keys())

    def _load(self, key):
        # return path
        return self.index_dict[key]

    # number of utterance
    def __len__(self):
        return len(self.index_dict)

    # avoid key error
    def __contains__(self, key):
        return key in self.index_dict

    # sequential index
    def __iter__(self):
        for key in self.index_keys:
            yield key, self._load(key)


class WaveReader(BaseReader):
    """
        Sequential/Random Reader for single channel wave
        Format of wav.scp follows Kaldi's definition:
            key1 /path/to/wav
            ...
    """
    def __init__(self, wav_scp, sr=16000, normalize=True):
        super(WaveReader, self).__init__(wav_scp)
        self.sr = sr
        self.normalize = normalize

    def _load(self, key):
        # return C x N or N
        sr, samps = read_wav(self.index_dict[key],
                             normalize=self.normalize,
                             return_rate=True)
        # if given samp_rate, check it
        if self.sr is not None and sr != self.sr:
            raise RuntimeError("Sample rate mismatch: {:d} vs {:d}".format(
                sr, self.sr))

        return samps


def read_wav(fname, beg=None, end=None, normalize=True, return_rate=False):
    """
    Read wave files using scipy.io.wavfile(support multi-channel)
    """
    # samps_int16: N x C or N
    #   N: number of samples
    #   C: number of channels
    if beg is not None:
        samps_int16, samp_rate = sf.read(fname,
                                         start=beg,
                                         stop=end,
                                         dtype="int16")
    else:
        samp_rate, samps_int16 = wf.read(fname)
    # N x C => C x N
    samps = samps_int16.astype(np.float32)
    # tranpose because I used to put channel axis first
    if samps.ndim != 1:
        samps = np.transpose(samps)
    # normalize like MATLAB and librosa
    if normalize:
        samps = samps / MAX_INT16
    if return_rate:
        return samp_rate, samps
    return samps


def write_wav(fname, samps, sr=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
    if normalize:
        samps = samps * MAX_INT16
    # scipy.io.wavfile.write could write single/multi-channel files
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
        samps = np.transpose(samps)
        samps = np.squeeze(samps)
    # same as MATLAB and kaldi
    samps_int16 = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir:
        os.makedirs(fdir, exist_ok=True)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wf.write(fname, sr, samps_int16)

