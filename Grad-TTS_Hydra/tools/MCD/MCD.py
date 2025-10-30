import argparse
import fnmatch
import logging
import multiprocessing as mp
import os
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pysptk
import soundfile as sf
from fastdtw import fastdtw
from scipy import spatial


def find_files(root_dir: str, query: List[str] = ["*.flac", "*.wav"]) -> List[str]: # return all files ending with the pattern specified in qeury in a directory 
    
    files = []
    if os.path.isfile(root_dir):
    # Case: root_dir is a file path
        for q in query:
            if fnmatch.fnmatch(root_dir, q):
                files.append(root_dir)
    else:
        for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
            for q in query:
                for filename in fnmatch.filter(filenames, q):
                    files.append(os.path.join(root, filename))
    return files


def sptk_extract(
    x: np.ndarray,
    fs: int,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
    is_padding: bool = False,
):
    """Extract SPTK-based mel-cepstrum.
    Args:
        x (ndarray): 1D waveform array.
        fs (int): Sampling rate
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).
        is_padding (bool): Whether to pad the end of signal (default=False).
    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).
    """
    # perform padding
    if is_padding:
        n_pad = n_fft - (len(x) - n_fft) % n_shift
        x = np.pad(x, (0, n_pad), "reflect")

    # get number of frames
    n_frame = (len(x) - n_fft) // n_shift + 1

    # get window function
    win = pysptk.sptk.hamming(n_fft)

    # check mcep and alpha
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)

    # calculate spectrogram
    mcep = [
        pysptk.mcep(
            x[n_shift * i : n_shift * i + n_fft] * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for i in range(n_frame)
    ]

    return np.stack(mcep)

def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]


def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")


def calculate(
    file_list: List[str],
    gt_file_list: List[str],
    args: argparse.Namespace,
    mcd_dict: Dict,
):
    """Calculate MCD."""
    for i, gen_path in enumerate(file_list):
        corresponding_list = list(
            filter(lambda gt_path: _get_basename(gt_path) == _get_basename(gen_path), gt_file_list)
        )
        assert len(corresponding_list) == 1
        gt_path = corresponding_list[0]
        gt_basename = _get_basename(gt_path)

        # load wav file as int16
        gen_x, gen_fs = sf.read(gen_path, dtype="float64")
        gt_x, gt_fs = sf.read(gt_path, dtype="float64")


        fs = gen_fs
        if gen_fs != gt_fs: raise ValueError("Sampling rate mismatch")

        
        
        # extract ground truth and converted features
        gen_mcep = sptk_extract(
            x=gen_x,
            fs=fs,
            n_fft=1024, #args.n_fft,
            n_shift=256, #args.n_shift,
            mcep_dim=None, #args.mcep_dim,
            mcep_alpha=None, #args.mcep_alpha,
        )
        gt_mcep = sptk_extract(
            x=gt_x,
            fs=fs,
            n_fft=1024,
            n_shift=256,
            mcep_dim=None, #args.mcep_dim,
            mcep_alpha=None, #args.mcep_alpha,
        )
        

        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        twf = np.array(path).T
        gen_mcep_dtw = gen_mcep[twf[0]]
        gt_mcep_dtw = gt_mcep[twf[1]]

        # MCD
        diff2sum = np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        mcd_dict[gt_basename] = mcd



def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate Mel-cepstrum distortion.")
    parser.add_argument(
        "--gt_wavdir_or_wavscp",
        type=str,
        help="Path of directory or wav.scp for ground truth waveforms.",
    )
    parser.add_argument(
        "--gen_wavdir_or_wavscp",
        type=str,
        help="Path of directory or wav.scp for generated waveforms.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path of directory to write the results.",
    )

    # analysis related
    parser.add_argument(
        "--mcep_dim",
        default=None,
        type=int,
        help=(
            "Dimension of mel cepstrum coefficients. "
            "If None, automatically set to the best dimension for the sampling."
        ),
    )
    parser.add_argument(
        "--mcep_alpha",
        default=None,
        type=float,
        help=(
            "All pass constant for mel-cepstrum analysis. "
            "If None, automatically set to the best dimension for the sampling."
        ),
    )
    parser.add_argument(
        "--n_fft",
        default=1024,
        type=int,
        help="The number of FFT points.",
    )
    parser.add_argument(
        "--n_shift",
        default=256,
        type=int,
        help="The number of shift points.",
    )
    parser.add_argument(
        "--nj",
        default=16,
        type=int,
        help="Number of parallel jobs.",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Run MCD calculation in parallel."""

    print('Starting a new Epoch Calculation')
    args = get_parser().parse_args()
    print(args.gen_wavdir_or_wavscp)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # find files
    gen_files = sorted(find_files(args.gen_wavdir_or_wavscp)) 
    gt_files = sorted(find_files(args.gt_wavdir_or_wavscp))

    logging.info("The number of generated utterances = %d" % len(gen_files))
    
    
    file_lists = np.array_split(gen_files, args.nj)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        mcd_dict = manager.dict()
        processes = []
        for f in file_lists:
            p = mp.Process(target=calculate, args=(f, gt_files, args, mcd_dict))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        # convert to standard list
        mcd_dict = dict(mcd_dict)

        # calculate statistics
        mean_mcd = np.mean(np.array([v for v in mcd_dict.values()]))
        std_mcd = np.std(np.array([v for v in mcd_dict.values()]))
        logging.info(f"Average: {mean_mcd:.4f} ± {std_mcd:.4f}")


    if not os.path.exists(f"{args.outdir}"):
        os.makedirs(f"{args.outdir}")

    with open(f"{args.outdir}/mean_mcd.txt", "a") as f:
        f.write(f"{mean_mcd:.4f}\n")

    with open(f"{args.outdir}/std_mcd.txt", "a") as f:
        f.write(f"{std_mcd:.4f}\n")
    
    with open(f"{args.outdir}/MCD.txt", "a") as f:
        f.write(f"{mean_mcd:.4f} \\pm {std_mcd:.4f}\n")

    logging.info("Successfully finished MCD evaluation.")


if __name__ == "__main__":
    main()
