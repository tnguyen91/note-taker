import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, effects
from scipy.signal import butter, lfilter
import webrtcvad
import struct
from typing import Union
from pydub.effects import compress_dynamic_range

def convert_to_standard(input_path: str, output_path: str, sr: int = 16000):
    """
    Convert any audio/video file to mono WAV @ `sr` Hz.
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(sr)
    audio.export(output_path, format="wav")


def band_pass_filter(data: np.ndarray, sr: int,
                     lowcut: float = 100.0,
                     highcut: float = None) -> np.ndarray:
    """
    Band-pass between lowcut and highcut (Hz). If highcut is None or >= Nyquist,
    clamp to 0.99 * Nyquist to satisfy SciPy’s 0 < Wn < 1 requirement.
    """
    nyq = sr / 2.0
    if highcut is None or highcut >= nyq:
        highcut = nyq * 0.99

    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(2, [low, high], btype='band')
    return lfilter(b, a, data)

def denoise_audio(input_path: str, output_path: str, sr: int = 16000):
    """
    Reduce noise via spectral subtraction (noisereduce).
    """
    # load
    data, _ = sf.read(input_path, dtype='float32')
    # assume first second is noise
    noise_clip = data[:sr]
    reduced = nr.reduce_noise(y=data, y_noise=noise_clip, sr=sr)
    # save
    sf.write(output_path, reduced, sr)


def normalize_volume(input_path: str, output_path: str) -> None:
    """
    Normalize loudness to -1.0…1.0 using pydub’s built-in normalize.
    """
    audio = AudioSegment.from_wav(input_path)
    normalized = effects.normalize(audio)
    normalized.export(output_path, format="wav")


def vad_filter(input_path: str, output_path: str, sr: int = 16000, aggressiveness: int = 2):
    """
    Keep only voiced frames (WebRTC VAD), stitch them into one stream.
    """
    # load raw PCM
    audio, _ = sf.read(input_path, dtype='int16')
    vad = webrtcvad.Vad(aggressiveness)

    frame_duration_ms = 30
    frame_length = int(sr * frame_duration_ms / 1000)
    voiced = []

    for i in range(0, len(audio) - frame_length, frame_length):
        frame = audio[i:i+frame_length]
        # pack into bytes for VAD
        is_speech = vad.is_speech(
            struct.pack('<' + ('h'*len(frame)), *frame),
            sample_rate=sr
        )
        if is_speech:
            voiced.append(frame)

    if voiced:
        voiced_audio = np.concatenate(voiced).astype(np.int16)
        # convert back to float32 for writing
        sf.write(output_path, voiced_audio.astype(np.float32) / 32768.0, sr)
    else:
        # fallback: export the original
        AudioSegment.from_wav(input_path).export(output_path, format="wav")

def compress_audio(input_path, output_path):
    seg = AudioSegment.from_wav(input_path)
    comp = compress_dynamic_range(seg, threshold=-20.0, ratio=4.0)
    comp.export(output_path, format="wav")
