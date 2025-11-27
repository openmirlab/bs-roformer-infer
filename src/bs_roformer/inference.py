"""CLI entry point for BS-Roformer inference."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from ml_collections import ConfigDict
from tqdm import tqdm

from .utils import demix_track, get_model_from_config


def _ensure_wav_inputs(input_folder: Path) -> list[Path]:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    wav_files = sorted(input_folder.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_folder}")
    return wav_files


def _resolve_output_dir(store_dir: Path) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def _format_iterable(paths: Iterable[Path], verbose: bool) -> Iterable[Path]:
    return tqdm(paths, desc="Tracks", unit="track") if not verbose else paths


def run_folder(model, args, config, device, verbose: bool = False) -> None:
    start_time = time.time()
    model.eval()

    input_folder = Path(args.input_folder).expanduser()
    store_dir = _resolve_output_dir(Path(args.store_dir).expanduser())
    all_mixtures_path = _ensure_wav_inputs(input_folder)
    total_tracks = len(all_mixtures_path)
    print(f"Total tracks found: {total_tracks}")

    instruments = config.training.instruments
    if getattr(config.training, "target_instrument", None) is not None:
        instruments = [config.training.target_instrument]

    iterable = _format_iterable(all_mixtures_path, verbose)

    first_chunk_time = None

    for track_number, path in enumerate(iterable, 1):
        print(f"\nProcessing track {track_number}/{total_tracks}: {path.name}")

        mix, sr = sf.read(path)
        original_mono = False
        if len(mix.shape) == 1:
            original_mono = True
            mix = np.stack([mix, mix], axis=-1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)

        if first_chunk_time is not None:
            total_length = mixture.shape[1]
            num_chunks = (total_length + config.inference.chunk_size // config.inference.num_overlap - 1) // (config.inference.chunk_size // config.inference.num_overlap)
            estimated_total_time = first_chunk_time * num_chunks
            print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds")
            sys.stdout.write(f"Estimated time remaining: {estimated_total_time:.2f} seconds\r")
            sys.stdout.flush()

        res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

        for instr in instruments:
            vocals_output = res[instr].T
            if original_mono:
                vocals_output = vocals_output[:, 0]

            vocals_path = store_dir / f"{path.stem}_{instr}.wav"
            sf.write(vocals_path, vocals_output, sr, subtype="FLOAT")

        if instruments:
            vocals_output = res[instruments[0]].T
            if original_mono:
                vocals_output = vocals_output[:, 0]

            original_mix, _ = sf.read(path)
            instrumental = original_mix - vocals_output

            instrumental_path = store_dir / f"{path.stem}_instrumental.wav"
            sf.write(instrumental_path, instrumental, sr, subtype="FLOAT")

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_folder(args):
    parser = argparse.ArgumentParser(description="BS-Roformer inference runner")
    parser.add_argument("--model_type", type=str, default="bs_roformer")
    parser.add_argument("--config_path", type=Path, required=True, help="path to config yaml file")
    parser.add_argument("--model_path", type=Path, required=True, help="path to the .ckpt weights")
    parser.add_argument("--input_folder", type=Path, required=True, help="folder with songs to process")
    parser.add_argument("--store_dir", type=Path, default=Path("outputs"), help="path to store model outputs")
    parser.add_argument("--device", type=str, default=None, help="torch device string, defaults to auto")
    parser.add_argument("--device_ids", nargs='+', type=int, help='optional list of gpu ids for DataParallel')
    if args is None:
        args = parser.parse_args()
    elif isinstance(args, argparse.Namespace):
        # Already parsed, use as is
        pass
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        config = ConfigDict(yaml.safe_load(f))

    model = get_model_from_config(args.model_type, config)
    print(f"Using model weights: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))

    device = _select_device(args)

    if args.device_ids:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for --device_ids usage")
        model = nn.DataParallel(model, device_ids=args.device_ids).to(device)
    else:
        model = model.to(device)

    run_folder(model, args, config, device, verbose=False)


def _select_device(args: argparse.Namespace) -> torch.device:
    if args.device:
        if args.device == "cpu":
            return torch.device("cpu")
        return torch.device(args.device)

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    print("CUDA is not available. Falling back to CPU. This will be slow.")
    return torch.device("cpu")


def main():
    """Main entry point for the BS-Roformer inference script."""
    proc_folder(None)


if __name__ == "__main__":
    main()
