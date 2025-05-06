import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, effects
import whisper
from scipy.signal import butter, sosfilt, buttord

def bandpass_filter(y, sr, lowcut=100, highcut=None):
    highcut = sr / 2.0 * 0.999
    sos = butter(N=5, Wn=[lowcut, highcut], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, y)

def preprocess(input_path, output_path="final3.wav", sr=16000, noise_duration=1.0):
    # load
    y, sr = librosa.load(input_path, sr=sr, mono=True)

    # Butterworth bandpass filter
    y = bandpass_filter(y, sr)

    sf.write(output_path, y, sr)
    
preprocess("sample.m4a")
