import os
import argparse
import tempfile
import soundfile as sf
import whisper
from preprocess_audio import preprocess 
from datetime import datetime

def chunk_audio(wav_path: str, chunk_length: float, overlap: float):
    y, sr = sf.read(wav_path, dtype='float32')
    step = int((chunk_length - overlap) * sr)
    size = int(chunk_length * sr)
    total = len(y)
    for start in range(0, total, step):
        end = min(start + size, total)
        chunk = y[start:end]
        offset = start / sr
        yield offset, chunk


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with whisper and preprocessing.")
    parser.add_argument('--input', '-i', required=True,
                        help="Path to raw input audio (wav, mp3, mp4, etc.)")
    parser.add_argument('--output', '-o', required=True,
                        help="Path to write transcript text file.")
    parser.add_argument('--model', default='medium',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help="Whisper model size.")
    parser.add_argument('--chunk_length', type=float, default=20.0,
                        help="Chunk length in seconds.")
    parser.add_argument('--overlap', type=float, default=0.5,
                        help="Overlap between chunks in seconds.")
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate for preprocessing.")
    parser.add_argument('--aggressiveness', type=int, default=2,
                        help="VAD aggressiveness (0-3).")
    parser.add_argument('--min_silence_ms', type=int, default=300,
                        help="Minimum silence (ms) to merge segments.")
    parser.add_argument('--padding_ms', type=int, default=50,
                        help="Padding (ms) around VAD segments.")
    args = parser.parse_args()

    # Preprocess 
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # create a temp file whose name starts with that timestamp, and won't auto-delete
    with tempfile.NamedTemporaryFile(prefix=f"preprocessed_{now}_", suffix=".wav", delete=False) as tmp: 
        tmp_wav = tmp.name
    preprocess(args.input, tmp_wav, sr=args.sr, aggressiveness=args.aggressiveness, frame_duration_ms=30, min_silence_ms=args.min_silence_ms, padding_ms=args.padding_ms)
    wav_path = tmp_wav

    # Load model
    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model)

    # Transcribe chunks
    segments = []
    for offset, chunk in chunk_audio(wav_path, args.chunk_length, args.overlap):
        if chunk.size == 0:
            continue

        result = model.transcribe(chunk, language='en', temperature=0.0)
        # result['segments'] is a list of {start, end, text}
        for seg in result.get('segments', []):
            segments.append({
                'start': seg['start'] + offset,
                'end': seg['end'] + offset,
                'text': seg['text'].strip()
            })

    # Sort and write transcript
    segments = sorted(segments, key=lambda x: x['start'])
    with open(args.output, 'w', encoding='utf-8') as f:
        for seg in segments:
            f.write(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}\n")

    print(f"Transcript written to {args.output}")

if __name__ == '__main__':
    main()
