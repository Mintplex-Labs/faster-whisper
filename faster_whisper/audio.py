from typing import Union, BinaryIO
import soundfile as sf
import numpy as np

def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    """
    Decodes the audio using soundfile. Assumes audio is pre-resampled to 16000 Hz mono channel.
    """
    audio, sr = sf.read(input_file, dtype="float32")

    if sr != sampling_rate:
         # WARNING: soundfile does NOT resample. If sr != 16000, the output will be incorrect.
         # Always assume the audio is already resampled to 16000 Hz mono channel.
         print(f"Warning: Audio sample rate is {sr} Hz, but expected {sampling_rate} Hz. Resampling is required for correctness.")

    if split_stereo:
        if audio.ndim == 2 and audio.shape[1] == 2:
            left_channel = audio[:, 0]
            right_channel = audio[:, 1]
            return left_channel, right_channel
        elif audio.ndim == 1:
            print("Warning: Attempted to split stereo, but audio is already mono.")
            return audio, audio

    # If the audio is stereo, convert it to mono by averaging the channels
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    return audio

def pad_or_trim(array, length: int = 3000, *, axis: int = -1):
    """
    Pad or trim the Mel features array to 3000, as expected by the encoder.
    """
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array