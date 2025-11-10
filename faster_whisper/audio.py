import numpy as np
import librosa
from typing import Union, BinaryIO, Tuple

# NOTE: The helper functions _ignore_invalid_frames, _group_frames, and 
# _resample_frames are no longer needed, as librosa.load handles decoding 
# and resampling in a single call.

def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Decodes and resamples the audio using librosa.

    Args:
      input_file: Path to the input file or a file-like object.
      sampling_rate: Resample the audio to this sample rate (default 16000 Hz).
      split_stereo: Return separate left and right channels.

    Returns:
      A float32 Numpy array (mono) or a 2-tuple of float32 Numpy arrays (stereo split).
    """
    mono_mode = True if not split_stereo else False
    audio_data, sr = librosa.load(
        input_file,
        sr=sampling_rate, 
        mono=mono_mode,   
        dtype=np.float32,
    )
    
    if split_stereo:
        if audio_data.ndim == 2 and audio_data.shape[0] >= 2:
            left_channel = audio_data[0, :]
            right_channel = audio_data[1, :]
            return left_channel, right_channel
            
        elif audio_data.ndim == 1:
            print("Warning: Attempted to split stereo, but audio source is mono.")
            return audio_data, audio_data
    return audio_data.flatten() 

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