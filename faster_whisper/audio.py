from typing import Union, BinaryIO
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly

def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    """
    Decodes and resamples the audio to sampling_rate (default 16000 Hz) 
    using soundfile and scipy.signal.
    """
    audio, sr = sf.read(input_file, dtype="float32")

    if sr != sampling_rate:
        print(f"Resampling audio from {sr} Hz to {sampling_rate} Hz.")
        if audio.ndim == 2:
            print("Audio is stereo, resampling both channels.")
            audio_resampled = np.zeros(
                (int(audio.shape[0] * sampling_rate / sr), 2), dtype=np.float32
            )
            for i in range(2):
                audio_resampled[:, i] = resample_poly(
                    audio[:, i], sampling_rate, sr
                ).astype(np.float32)
            audio = audio_resampled
        else:
            audio = resample_poly(audio, sampling_rate, sr).astype(np.float32)

    if split_stereo:
        if audio.ndim == 2 and audio.shape[1] == 2:
            left_channel = audio[:, 0]
            right_channel = audio[:, 1]
            return left_channel, right_channel
        elif audio.ndim == 1:
            print("Warning: Attempted to split stereo, but audio is mono.")
            return audio, audio

    # Convert stereo to mono by averaging the channels
    if audio.ndim == 2: audio = audio.mean(axis=1)

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