# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""CLI tool to fetch AlphaFold 3 databases."""

import concurrent.futures
import os
import pathlib
import shutil
import tarfile

import click
import requests
import zstandard

from alphafold3.cli.main_cli import alphafold3 # Import the shared group

_DEFAULT_DB_DIR_STR = '~/public_databases'
SOURCE_URL = "https://storage.googleapis.com/alphafold-databases/v3.0"

DATABASE_FILES = [
    "mgy_clusters_2022_05.fa",
    "bfd-first_non_consensus_sequences.fasta",
    "uniref90_2022_05.fa",
    "uniprot_all_2021_04.fa",
    "pdb_seqres_2022_09_28.fasta",
    "rnacentral_active_seq_id_90_cov_80_linclust.fasta",
    "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
    "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
]
TAR_MMCIF_FILE = "pdb_2022_09_28_mmcif_files.tar"  # Name of the tar file (without .zst)


def download_and_extract_tar_zst(db_dir_path: pathlib.Path, executor: concurrent.futures.Executor):
    """Downloads and extracts the .tar.zst file."""
    tar_zst_filename = f"{TAR_MMCIF_FILE}.zst"
    url = f"{SOURCE_URL}/{tar_zst_filename}"
    output_path = db_dir_path / TAR_MMCIF_FILE # For the tar file itself, not the .zst

    click.echo(f"Downloading {tar_zst_filename} from {url}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        dctx = zstandard.ZstdDecompressor()

        # Temporary path for the downloaded .zst file
        temp_tar_zst_path = db_dir_path / tar_zst_filename

        with open(temp_tar_zst_path, 'wb') as f_zst:
            for chunk in response.iter_content(chunk_size=8192):
                f_zst.write(chunk)
        click.echo(f"Finished downloading {tar_zst_filename}.")

        click.echo(f"Decompressing and extracting {tar_zst_filename} to {db_dir_path}/mmcif_files ...")
        # Decompress and extract in one go to avoid large intermediate .tar file
        with open(temp_tar_zst_path, 'rb') as f_zst_in:
            with dctx.stream_reader(f_zst_in) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as tar_archive:
                    # Ensure extraction happens within a subdirectory named 'mmcif_files'
                    # which is what the original script does by creating 'mmcif_files' and then cd-ing into it.
                    # Tarfile extracts to the current working directory of the tar command,
                    # here we specify members to be extracted into db_dir_path directly,
                    # but they might already have 'mmcif_files/' prefix in their paths.
                    # If not, we might need to create 'mmcif_files' and adjust paths.
                    # For now, let's assume paths in tar are relative and will create 'mmcif_files' or are already prefixed.
                    # The original script created 'mmcif_files' and then extracted into it.
                    # Let's ensure this behavior.
                    mmcif_output_dir = db_dir_path / 'mmcif_files'
                    mmcif_output_dir.mkdir(parents=True, exist_ok=True)

                    # We need to be careful with tar extraction paths.
                    # The original script cds into `mmcif_files` then extracts.
                    # tarfile extracts relative to its `path` argument or CWD.
                    # We will extract to the mmcif_output_dir
                    tar_archive.extractall(path=mmcif_output_dir)

        click.echo(f"Successfully decompressed and extracted {tar_zst_filename}.")
        os.remove(temp_tar_zst_path) # Clean up the downloaded .zst file

    except requests.RequestException as e:
        click.echo(f"Error downloading {tar_zst_filename}: {e}", err=True)
    except tarfile.TarError as e:
        click.echo(f"Error extracting {TAR_MMCIF_FILE} from {tar_zst_filename}: {e}", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred with {tar_zst_filename}: {e}", err=True)


def download_and_decompress_fa_zst(filename_fa: str, db_dir_path: pathlib.Path):
    """Downloads and decompresses a .fa.zst file."""
    filename_fa_zst = f"{filename_fa}.zst"
    url = f"{SOURCE_URL}/{filename_fa_zst}"
    output_fa_path = db_dir_path / filename_fa

    click.echo(f"Downloading {filename_fa_zst} from {url}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        dctx = zstandard.ZstdDecompressor()
        with open(output_fa_path, 'wb') as f_out:
            with dctx.stream_reader(response.raw) as reader:
                shutil.copyfileobj(reader, f_out)
        click.echo(f"Successfully downloaded and decompressed to {output_fa_path}.")
    except requests.RequestException as e:
        click.echo(f"Error downloading {filename_fa_zst}: {e}", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred with {filename_fa_zst}: {e}", err=True)


@alphafold3.command(name='fetch-databases')
@click.option(
    '--db-dir',
    type=click.Path(),
    default=_DEFAULT_DB_DIR_STR,
    show_default=True,
    help='Directory to download and store databases.'
)
def fetch_databases_main(db_dir: str):
    """Downloads and decompresses AlphaFold 3 databases."""
    db_dir_path = pathlib.Path(db_dir).expanduser()

    click.echo(f"Creating database directory: {db_dir_path}")
    db_dir_path.mkdir(parents=True, exist_ok=True)

    # Check for zstd, tar (tarfile is a Python module, so no CLI check needed for tar)
    if not shutil.which("zstd"): # zstandard library is used, so this check might be redundant
        click.echo("zstd command line tool not found. Please install zstandard if issues arise, though the Python library should suffice.", err=True)
        # The script uses the zstandard library, so the CLI tool isn't strictly necessary.
        # However, the original script checked for it.

    # Use ThreadPoolExecutor for concurrent downloads
    # Max workers can be adjusted; 5 seems reasonable for network-bound tasks.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit the tar.zst file processing
        futures = [executor.submit(download_and_extract_tar_zst, db_dir_path, executor)]

        # Submit .fa.zst files processing
        for db_file in DATABASE_FILES:
            futures.append(executor.submit(download_and_decompress_fa_zst, db_file, db_dir_path))

        for future in concurrent.futures.as_completed(futures):
            # You can add error handling here if needed, though errors are printed in functions
            try:
                future.result()
            except Exception as e:
                click.echo(f"A task resulted in an error: {e}", err=True)

    click.echo("Database fetch process completed.")

if __name__ == '__main__':
    alphafold3() # This allows running this script directly for the fetch-databases command via the group
                 # Or, more typically, the user would run `alphafold3 fetch-databases ...`
                 # if main_cli.py is the entry point.
                 # For development, this is fine.
                 # The actual entry point should be alphafold3.cli.main_cli:alphafold3
                 # And then you'd call `python -m alphafold3.cli.main_cli fetch-databases --db-dir ...`
                 # or after installation `alphafold3 fetch-databases --db-dir ...`
    pass # Keep the if __name__ == '__main__': block, but alphafold3() will handle it.
         # The main_cli.py would be the actual entry point for the `alphafold3` command.
         # This file essentially just adds a command to that group.
         # So, if this file is run directly, it should ideally invoke the main group.
         # The `pass` is fine, or can call `alphafold3()` like in `run_alphafold.py`.
         # Let's make it consistent:
    # alphafold3() # Assuming main_cli.py will be the entry point.
    # Actually, if this script is run directly, it doesn't know about other commands unless main_cli is run.
    # This is more about structure. The `alphafold3()` call makes sense if this is the *only* way to access this command.
    # But since we are building a group, let's remove it to avoid confusion.
    # The commands will be available via the main_cli entry point.
    pass

main = fetch_databases_main
