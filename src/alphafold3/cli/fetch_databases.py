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
from typing import Optional
from alphafold3.config.paths import get_database_dirs

import click
import requests
import zstandard

from alphafold3.cli.main_cli import alphafold3 # Import the shared group

# _DEFAULT_DB_DIR_STR = '~/public_databases'
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
    "db_dir_override",  # Renames the argument in the function signature
    type=click.Path(file_okay=False, resolve_path=True),
    default=None,
    show_default=False,
    help='Specific directory to download/store databases. Overrides configured paths. '
         'If not set, attempts to use paths from AF3_DB_DIRS, config file, or defaults like ~/.alphafold3/databases.'
)
def fetch_databases_main(db_dir_override: Optional[str]):
    """Downloads and decompresses AlphaFold 3 databases."""
    target_db_dir_path: Optional[pathlib.Path] = None

    if db_dir_override:
        # User provided a specific directory for this download operation
        target_db_dir_path = pathlib.Path(db_dir_override).expanduser().resolve()
        click.echo(f"Using specified download directory: {target_db_dir_path}")
    else:
        # No specific directory given, try to get from config system
        try:
            # We pass None to cli_db_dirs as we are not using CLI args for this internal lookup
            configured_db_dirs = get_database_dirs(cli_db_dirs=None)

            # Try to find the first writable directory among configured paths
            for d_path in configured_db_dirs:
                # Check if path exists and is writable
                if d_path.exists() and d_path.is_dir() and os.access(d_path, os.W_OK):
                    target_db_dir_path = d_path
                    click.echo(f"Using first writable configured database directory for download: {target_db_dir_path}")
                    break
                # If it doesn't exist, check if parent is writable to allow creation
                elif not d_path.exists() and d_path.parent.exists() and os.access(d_path.parent, os.W_OK):
                    target_db_dir_path = d_path
                    click.echo(f"Using configured database directory (will be created): {target_db_dir_path}")
                    break

            if not target_db_dir_path and configured_db_dirs:
                # If configured_db_dirs were found but none were suitable (e.g. all read-only mounts)
                # Use the first configured path; mkdir will test writability.
                target_db_dir_path = configured_db_dirs[0]
                click.echo(f"Using first configured database directory (writability will be tested by mkdir): {target_db_dir_path}")

        except FileNotFoundError:
            # This exception from get_database_dirs means no paths were resolved.
            # Will be handled by the final fallback.
            pass

        if not target_db_dir_path:
            # Final fallback if no path found from override, config, or writable defaults
            primary_download_default = pathlib.Path.home() / '.alphafold3' / 'databases'
            target_db_dir_path = primary_download_default
            click.echo(f"No suitable configured directory found or no configuration set. "
                       f"Defaulting download to: {target_db_dir_path}")

    if not target_db_dir_path: # Should be practically unreachable
        click.echo("Critical Error: Could not determine a download directory.", err=True)
        return

    click.echo(f"Ensuring database directory exists and is writable: {target_db_dir_path}")
    try:
        target_db_dir_path.mkdir(parents=True, exist_ok=True)
        # Test writability explicitly
        test_file = target_db_dir_path / ".af3_writetest"
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        click.echo(f"Error: Target database directory {target_db_dir_path} is not writable or cannot be created: {e}", err=True)
        click.echo("Please check permissions or specify a writable directory using --db-dir.")
        return

    if not shutil.which("zstd"):
        click.echo("Warning: zstd command line tool not found. The Python zstandard library will be used.", dim=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures_list = [executor.submit(download_and_extract_tar_zst, target_db_dir_path, executor)]
        for db_file in DATABASE_FILES:
            futures_list.append(executor.submit(download_and_decompress_fa_zst, db_file, target_db_dir_path))

        completed_count = 0
        total_tasks = len(futures_list)
        for future_item in concurrent.futures.as_completed(futures_list):
            try:
                future_item.result()
                completed_count +=1
                click.echo(f"Completed {completed_count}/{total_tasks} download/extraction tasks.")
            except Exception as e:
                click.echo(f"A download/extraction task resulted in an error: {e}", err=True)

        if completed_count == total_tasks:
            click.echo(f"All {total_tasks} database fetch tasks completed successfully. Files are in {target_db_dir_path}.")
        else:
            click.echo(f"Database fetch process completed with {total_tasks - completed_count} error(s) out of {total_tasks} tasks. Check logs above.", err=True)

if __name__ == '__main__':
    pass
