"""Model construction from config + chunked windowed inference (overlap-add demixing).

get_model_from_config filters a raw training-config dict down to the keys BSRoformer's
constructor actually accepts (training configs carry extra fields BSRoformer doesn't
take) and converts list-typed params back to the tuples the type hints require.
demix_track splits long mixtures into overlapping chunks, applies a linear
fade-in/out window per chunk to avoid audible seams at chunk boundaries, and
normalizes the result by the accumulated window weight -- this is what lets
inference run on audio far longer than a single forward pass could hold in memory.

Reads: .bs_roformer.BSRoformer, torch
"""

import time
import numpy as np
import torch
import sys
import torch.nn as nn


def get_model_from_config(model_type, config):
    if model_type == 'bs_roformer':
        from . import BSRoformer
        model_config = dict(config.model)

        # Convert list to tuple for parameters that require tuple type hints
        tuple_params = ['multi_stft_resolutions_window_sizes', 'freqs_per_bands']
        for param in tuple_params:
            if param in model_config and isinstance(model_config[param], list):
                model_config[param] = tuple(model_config[param])

        # Filter out parameters not accepted by BSRoformer
        valid_params = {
            'dim', 'depth', 'stereo', 'num_stems', 'time_transformer_depth',
            'freq_transformer_depth', 'freqs_per_bands', 'freq_range', 'dim_head',
            'heads', 'attn_dropout', 'ff_dropout', 'flash_attn', 'num_residual_streams',
            'num_residual_fracs', 'dim_freqs_in', 'stft_n_fft', 'stft_hop_length',
            'stft_win_length', 'stft_normalized', 'zero_dc', 'stft_window_fn',
            'mask_estimator_depth', 'multi_stft_resolution_loss_weight',
            'multi_stft_resolutions_window_sizes', 'multi_stft_hop_size',
            'multi_stft_normalized', 'multi_stft_window_fn'
        }
        model_config = {k: v for k, v in model_config.items() if k in valid_params}

        model = BSRoformer(**model_config)
    else:
        print('Unknown model: {}'.format(model_type))
        model = None

    return model


def get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)

def demix_track(config, model, mix, device, first_chunk_time=None):
    # chunk_size can be in inference or audio section depending on config version
    if hasattr(config.inference, 'chunk_size'):
        C = config.inference.chunk_size
    elif hasattr(config, 'audio') and hasattr(config.audio, 'chunk_size'):
        C = config.audio.chunk_size
    else:
        C = 588800  # default chunk size
    N = config.inference.num_overlap
    step = C // N
    fade_size = C // 10
    border = C - step

    if mix.shape[1] > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode='reflect')

    windowing_array = get_windowing_array(C, fade_size, device)

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)

            i = 0
            total_length = mix.shape[1]
            num_chunks = (total_length + step - 1) // step

            if first_chunk_time is None:
                start_time = time.time()
                first_chunk = True
            else:
                start_time = None
                first_chunk = False

            while i < total_length:
                part = mix[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                    else:
                        part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

                if first_chunk and i == 0:
                    chunk_start_time = time.time()

                x = model(part.unsqueeze(0))[0]

                window = windowing_array.clone()
                if i == 0:
                    window[:fade_size] = 1
                elif i + C >= total_length:
                    window[-fade_size:] = 1

                result[..., i:i+length] += x[..., :length] * window[..., :length]
                counter[..., i:i+length] += window[..., :length]
                i += step

                if first_chunk and i == step:
                    chunk_time = time.time() - chunk_start_time
                    first_chunk_time = chunk_time
                    estimated_total_time = chunk_time * num_chunks
                    print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds")
                    first_chunk = False

                if first_chunk_time is not None and i > step:
                    chunks_processed = i // step
                    time_remaining = first_chunk_time * (num_chunks - chunks_processed)
                    sys.stdout.write(f"\rEstimated time remaining: {time_remaining:.2f} seconds")
                    sys.stdout.flush()

            print()
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if mix.shape[1] > 2 * border and border > 0:
                estimated_sources = estimated_sources[..., border:-border]

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}, first_chunk_time
    else:
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}, first_chunk_time
