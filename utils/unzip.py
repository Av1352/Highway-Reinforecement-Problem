import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

def decompress_gz_file(gz_file_path, output_file_path):
    with gzip.open(gz_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def decompress_all_gz_files(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for folder in input_path.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            replay_logs_folder = folder / "replay_logs"

            if replay_logs_folder.exists():
                output_folder = output_path / folder_name
                output_folder.mkdir(parents=True, exist_ok=True)

            for gz_file in tqdm(replay_logs_folder.glob("*.gz"), desc=f"Decompressing {folder_name}"):
                output_file = output_folder / gz_file.stem
                decompress_gz_file(gz_file, output_file)


def decompress_some_gz_files(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    for gz_file in input_path.glob("*.gz"):
        output_file = output_path / gz_file.stem
        decompress_gz_file(gz_file, output_file)


# if __name__ == "__main__":
#     input_dir = "data/Pong"
#     output_dir = "data/PongData"
#
#     decompress_all_gz_files(input_dir, output_dir)

if __name__ == "__main__":
    input_dir = "data/temp"
    output_dir = "data/PongData"

    decompress_some_gz_files(input_dir, output_dir)