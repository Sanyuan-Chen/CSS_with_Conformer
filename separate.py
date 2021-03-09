#!/usr/bin/env python

import yaml
import argparse

import torch as th
import numpy as np

from pathlib import Path

from nnet import supported_nnet
from executor.executor import Executor
from utils.audio_util import WaveReader, write_wav
from utils.mvdr_util import make_mvdr


class EgsReader(object):
    """
    Egs reader
    """
    def __init__(self,
                 mix_scp,
                 sr=16000):
        self.mix_reader = WaveReader(mix_scp, sr=sr)

    def __len__(self):
        return len(self.mix_reader)

    def __iter__(self):
        for key, mix in self.mix_reader:
            egs = dict()
            egs["mix"] = mix
            yield key, egs


class Separator(object):
    """
    A simple wrapper for speech separation
    """
    def __init__(self, cpt_dir, get_mask=False, device_id=-1):
        # load executor
        cpt_dir = Path(cpt_dir)
        self.get_mask = get_mask
        self.executor = self._load_executor(cpt_dir)
        cpt_ptr = cpt_dir / "best.pt.tar"
        epoch = self.executor.resume(cpt_ptr.as_posix())
        print(f"Load checkpoint at {cpt_dir}, on epoch {epoch}")
        print(f"Nnet summary: {self.executor}")
        if device_id < 0:
            self.device = th.device("cpu")
        else:
            self.device = th.device(f"cuda:{device_id:d}")
            self.executor.to(self.device)
        self.executor.eval()

    def separate(self, egs):
        """
        Do separation
        """
        egs["mix"] = th.from_numpy(egs["mix"][None, :]).to(self.device, non_blocking=True)
        with th.no_grad():
            spks = self.executor(egs)
            spks = [s.detach().squeeze().cpu().numpy() for s in spks]
            return spks

    def _load_executor(self, cpt_dir):
        """
        Load executor from checkpoint
        """
        with open(cpt_dir / "train.yaml", "r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        nnet_type = conf["nnet_type"]
        if nnet_type not in supported_nnet:
            raise RuntimeError(f"Unknown network type: {nnet_type}")
        nnet = supported_nnet[nnet_type](**conf["nnet_conf"])
        executor = Executor(nnet, extractor_kwargs=conf["extractor_conf"], get_mask=self.get_mask)
        return executor



def run(args):
    # egs reader
    egs_reader = EgsReader(args.mix_scp, sr=args.sr)
    # separator
    seperator = Separator(args.checkpoint, device_id=args.device_id, get_mask=args.mvdr)

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)

    print(f"Start Separation " + ("w/ mvdr" if args.mvdr else "w/o mvdr"))
    for key, egs in egs_reader:
        print(f"Processing utterance {key}...")
        mixed = egs["mix"]
        spks = seperator.separate(egs)

        if args.mvdr:
            res1, res2 = make_mvdr(np.asfortranarray(mixed.T), spks)
            spks = [res1, res2]

        for i, s in enumerate(spks):
            if i < args.num_spks:
                write_wav(dump_dir / f"{key}_{i}.wav",
                          s * 0.9 / np.max(np.abs(s)))

    print(f"Processed {len(egs_reader)} utterances done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do speech separation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument("--mix-scp",
                        type=str,
                        required=True,
                        help="Rspecifier for mixed audio")
    parser.add_argument("--num_spks",
                        type=int,
                        default=2,
                        help="Number of the speakers")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, -1 means "
                        "running on CPU")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate for mixture input")
    parser.add_argument("--dump-dir",
                        type=str,
                        default="sep",
                        help="Directory to dump separated speakers")
    parser.add_argument("--mvdr",
                        type=bool,
                        default=False,
                        help="apply mvdr")
    args = parser.parse_args()
    run(args)