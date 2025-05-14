import os
from pydub import AudioSegment
import tempfile
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, lfilter
import webrtcvad
import struct
from nara_wpe.wpe import wpe

# Convert to mono WAV @ target sample rate
def convert_to_wav(input_path: str, sr: int = 16000):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(sr)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    audio.export(tmp.name, format='wav')
    tmp.close()
    return tmp.name

def pre_emphasis(x, coeff=0.97):
    # high-pass boosting high freqs
    return np.append(x[0], x[1:] - coeff * x[:-1])

# Band-pass filter to isolate speech
def band_pass_filter(y: np.ndarray, sr: int, lowcut: float = 120.0, highcut: float = 8000.0):
    nyq = 0.5 * sr
    b, a = butter(4, [lowcut / nyq, highcut / nyq * 0.999], btype='band')
    return lfilter(b, a, y)

# Spectral noise reduction using noisereduce
def denoise_spectral(y: np.ndarray, sr: int):
    noise_clip = y[:sr]
    return nr.reduce_noise(y=y, y_noise=noise_clip, sr=sr)

# VAD-based trimming with merge & padding
def vad_segments(y: np.ndarray, sr: int, aggressiveness: int = 2, frame_duration_ms: int = 30, min_silence_ms: int = 300, padding_ms: int = 50):
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * frame_duration_ms / 1000)
    num_frames = len(y) // frame_len

    # Frame-level speech detection
    flags = []
    for i in range(num_frames):
        frame = y[i * frame_len:(i + 1) * frame_len]
        pcm = struct.pack('<' + 'h' * len(frame), *np.int16(frame * 32767))
        flags.append(vad.is_speech(pcm, sr))

    # Identify contiguous speech segments
    segments = []
    seg_start = None
    for i, flag in enumerate(flags):
        if flag and seg_start is None:
            seg_start = i
        if seg_start is not None and (not flag or i == num_frames - 1):
            seg_end = i if flag else i - 1
            segments.append((seg_start, seg_end))
            seg_start = None

    # Merge segments with short gaps
    gap_thresh = int(min_silence_ms / frame_duration_ms)
    merged = []
    for start, end in segments:
        if not merged:
            merged.append((start, end))
        else:
            prev_s, prev_e = merged[-1]
            if start - prev_e - 1 <= gap_thresh:
                merged[-1] = (prev_s, end)
            else:
                merged.append((start, end))

    # Apply padding around segments
    pad = int(padding_ms / frame_duration_ms)
    padded = [(
        max(0, s - pad),
        min(num_frames - 1, e + pad)
    ) for s, e in merged]

    # Stitch frames
    chunks = [y[s * frame_len:(e + 1) * frame_len] for s, e in padded]
    return np.concatenate(chunks) if chunks else y

# Main function
def preprocess(input_path: str, output_path: str, sr: int = 16000, aggressiveness: int = 2, frame_duration_ms: int = 30, min_silence_ms: int = 300, padding_ms: int = 50):
    wav_path = convert_to_wav(input_path, sr)
    y, fs = sf.read(wav_path, dtype='float32')

    y = pre_emphasis(y, coeff=0.97)

    y = denoise_spectral(y, fs)

    y = band_pass_filter(y, fs)

    y = wpe(y[np.newaxis], taps=5, delay=5)[0]

    y = y / np.max(np.abs(y)) * 0.8
    
    y = vad_segments(y, fs, aggressiveness=aggressiveness, frame_duration_ms=frame_duration_ms, min_silence_ms=min_silence_ms, padding_ms=padding_ms)

    sf.write(output_path, y, fs)
    os.remove(wav_path)
    print(f"Preprocessed audio saved at: {output_path}")

