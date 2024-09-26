import os
import argparse
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, List
import sys
from pydub import AudioSegment

def parse_stm_file(stm_path: str) -> List[Dict]:
    """
    Parses a single STM file into a list of transcription entries.

    Args:
        stm_path (str): Path to the STM file.

    Returns:
        List[Dict]: List of dictionaries containing 'start_time', 'end_time', and 'transcription'.
    """
    entries = []
    try:
        with open(stm_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(';;') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                start_time = float(parts[3])
                end_time = float(parts[4])
                transcription = ' '.join(parts[6:])
                entries.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'transcription': transcription
                })
    except Exception as e:
        logging.error(f"Error parsing STM file {stm_path}: {e}")
    return entries

def chunk_transcripts(entries: List[Dict], segment_starts: List[float], segment_ends: List[float]) -> List[str]:
    """
    Chunks transcriptions into segments based on provided start and end times.

    Args:
        entries (List[Dict]): List of transcription entries.
        segment_starts (List[float]): List of segment start times in seconds.
        segment_ends (List[float]): List of segment end times in seconds.

    Returns:
        List[str]: List of segmented transcriptions.
    """
    segments = []
    for start, end in zip(segment_starts, segment_ends):
        segment_text = ''
        for entry in entries:
            # Check if transcription overlaps with the current segment
            if entry['end_time'] > start and entry['start_time'] < end:
                segment_text += ' ' + entry['transcription']
        segments.append(segment_text.strip())
    return segments

def get_audio_segments_info(audio_path: str, segment_length_ms: int, overlap_ms: int):
    """
    Computes the start and end times for audio segments.

    Args:
        audio_path (str): Path to the audio file.
        segment_length_ms (int): Length of each segment in milliseconds.
        overlap_ms (int): Overlap between segments in milliseconds.

    Returns:
        Tuple[List[float], List[float]]: Lists of segment start times and end times in seconds.
    """
    audio = AudioSegment.from_wav(audio_path)
    duration_ms = len(audio)
    segment_starts = []
    segment_ends = []

    start = 0
    while start < duration_ms:
        end = start + segment_length_ms
        if end > duration_ms:
            end = duration_ms
        segment_starts.append(start / 1000.0)  # Convert to seconds
        segment_ends.append(end / 1000.0)      # Convert to seconds
        start += segment_length_ms - overlap_ms

    return segment_starts, segment_ends

def process_single_stm(stm_file: str, stm_dir: str, audio_dir: str, output_dir: str,
                       segment_length_ms: int, overlap_ms: int) -> None:
    """
    Processes a single STM file: parses it, chunks transcriptions, and writes to output.

    Args:
        stm_file (str): Filename of the STM file.
        stm_dir (str): Directory containing STM files.
        audio_dir (str): Directory containing the corresponding audio files.
        output_dir (str): Directory to save segmented transcripts.
        segment_length_ms (int): Length of each segment in milliseconds.
        overlap_ms (int): Overlap between segments in milliseconds.
    """
    stm_path = os.path.join(stm_dir, stm_file)
    audio_filename = os.path.splitext(stm_file)[0] + '.wav'
    audio_path = os.path.join(audio_dir, audio_filename)

    if not os.path.exists(audio_path):
        logging.warning(f"Audio file {audio_filename} not found for STM file {stm_file}. Skipping.")
        return

    try:
        entries = parse_stm_file(stm_path)
        if not entries:
            logging.warning(f"No valid entries found in {stm_file}. Skipping.")
            return

        # Get segment start and end times matching the audio chunking
        segment_starts, segment_ends = get_audio_segments_info(audio_path, segment_length_ms, overlap_ms)

        # Chunk transcriptions based on segment times
        segments = chunk_transcripts(entries, segment_starts, segment_ends)

        # Write each segment to a separate file, matching the audio segment filenames
        base_filename = os.path.splitext(stm_file)[0]
        for idx, segment in enumerate(segments):
            segment_filename = f"{base_filename}_segment{idx + 1}.txt"
            segment_path = os.path.join(output_dir, segment_filename)
            with open(segment_path, 'w', encoding='utf-8') as f:
                f.write(segment)
        logging.info(f'Processed {stm_file} into {len(segments)} segments.')
    except Exception as e:
        logging.error(f"Failed to process {stm_file}: {e}")

def setup_logging(verbose: bool) -> None:
    """
    Configures the logging settings.

    Args:
        verbose (bool): If True, set logging level to INFO. Else, WARNING.
    """
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def process_stm_files(stm_dir: str, audio_dir: str, output_dir: str,
                      segment_length_ms: int = 20000, overlap_ms: int = 5000,
                      num_workers: int = None) -> None:
    """
    Processes all STM files in the specified directory using multiprocessing.

    Args:
        stm_dir (str): Directory containing STM files.
        audio_dir (str): Directory containing the corresponding audio files.
        output_dir (str): Directory to save segmented transcripts.
        segment_length_ms (int, optional): Length of each segment in milliseconds. Defaults to 20000.
        overlap_ms (int, optional): Overlap between segments in milliseconds. Defaults to 5000.
        num_workers (int, optional): Number of worker processes. Defaults to number of CPU cores minus one.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory at {output_dir}.")
    else:
        logging.info(f"Output directory already exists at {output_dir}.")

    stm_files = [f for f in os.listdir(stm_dir) if f.lower().endswith('.stm')]
    if not stm_files:
        logging.warning(f"No STM files found in {stm_dir}. Exiting.")
        return

    logging.info(f"Found {len(stm_files)} STM files to process.")

    if not num_workers:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free

    logging.info(f"Using {num_workers} worker processes for multiprocessing.")

    # Partial function with fixed parameters except for stm_file
    worker = partial(
        process_single_stm,
        stm_dir=stm_dir,
        audio_dir=audio_dir,
        output_dir=output_dir,
        segment_length_ms=segment_length_ms,
        overlap_ms=overlap_ms
    )

    with Pool(processes=num_workers) as pool:
        # Use tqdm for progress bar
        from tqdm import tqdm
        for _ in tqdm(pool.imap_unordered(worker, stm_files), total=len(stm_files), desc="Processing STM files"):
            pass  # imap_unordered already handles the processing and side effects

    logging.info("Completed processing all STM files.")

def main():
    parser = argparse.ArgumentParser(description="Chunk STM Files into Segments matching Audio Chunks")
    parser.add_argument(
        "--stm_dir",
        type=str,
        required=True,
        help="Directory containing STM files."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing the corresponding audio files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save segmented transcripts."
    )
    parser.add_argument(
        "--segment_length_ms",
        type=int,
        default=20000,
        help="Length of each segment in milliseconds. Default is 20000 (20 seconds)."
    )
    parser.add_argument(
        "--overlap_ms",
        type=int,
        default=5000,
        help="Overlap between segments in milliseconds. Default is 5000 (5 seconds)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to number of CPU cores minus one."
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Increase output verbosity."
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    process_stm_files(
        stm_dir=args.stm_dir,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        segment_length_ms=args.segment_length_ms,
        overlap_ms=args.overlap_ms,
        num_workers=args.num_workers
    )

if __name__ == '__main__':
    main()
