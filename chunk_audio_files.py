import os
from pydub import AudioSegment
from multiprocessing import Pool, cpu_count
from functools import partial

def chunk_audio_file(wav_file, wav_dir, output_dir, segment_length_ms, overlap_ms):
    wav_path = os.path.join(wav_dir, wav_file)
    audio = AudioSegment.from_wav(wav_path)
    duration = len(audio)
    start = 0
    segment_num = 1
    while start < duration:
        end = start + segment_length_ms
        if end > duration:
            end = duration
        segment = audio[start:end]
        segment_filename = f"{os.path.splitext(wav_file)[0]}_segment{segment_num}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        segment.export(segment_path, format='wav')
        print(f'Created segment {segment_filename}')
        start += segment_length_ms - overlap_ms
        segment_num += 1

def chunk_audio_files_multiprocessing(wav_dir, output_dir, segment_length=20000, overlap=5000, num_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]

    # Parameters for processing
    segment_length_ms = segment_length
    overlap_ms = overlap

    # Use multiprocessing Pool
    pool = Pool(processes=num_workers)
    func = partial(chunk_audio_file, wav_dir=wav_dir, output_dir=output_dir,
                   segment_length_ms=segment_length_ms, overlap_ms=overlap_ms)

    pool.map(func, wav_files)
    pool.close()
    pool.join()

if __name__ == '__main__':
    wav_dir = 'E:\TED\WAV'       # Directory with WAV files from Stage 1
    output_dir = 'E:\TED\AUDIO CHUNKS' # Output directory for segments
    num_workers = 4             # Number of worker processes
    segment_length = 20000      # Segment length in milliseconds (20 seconds)
    overlap = 5000              # Overlap in milliseconds (5 seconds)

    chunk_audio_files_multiprocessing(wav_dir, output_dir, segment_length, overlap, num_workers)
