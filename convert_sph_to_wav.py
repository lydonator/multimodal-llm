import os
import subprocess

def convert_sph_to_wav(sph_dir, wav_dir):
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    sph_files = [f for f in os.listdir(sph_dir) if f.endswith('.sph')]
    for sph_file in sph_files:
        sph_path = os.path.join(sph_dir, sph_file)
        wav_file = os.path.splitext(sph_file)[0] + '.wav'
        wav_path = os.path.join(wav_dir, wav_file)
        print(f'Converting {sph_file} to {wav_file}')
        try:
            subprocess.run(['sox', sph_path, wav_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {sph_file}: {e}")

if __name__ == '__main__':
    sph_dir = 'E:\TED\SPH'  # Replace with your SPH files directory
    wav_dir = 'E:\TED\WAV'  # Output directory for WAV files
    convert_sph_to_wav(sph_dir, wav_dir)
