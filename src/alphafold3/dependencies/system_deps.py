# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

import shutil
import click

HMMER_BINARIES = [
    "jackhmmer",
    "nhmmer",
    "hmmalign",
    "hmmsearch",
    "hmmbuild",
]

def check_hmmer_dependencies():
    """Checks if all HMMER binaries are available in the system PATH.

    Raises:
        click.ClickException: If any HMMER binaries are not found.
    """
    missing_binaries = []
    for binary in HMMER_BINARIES:
        if shutil.which(binary) is None:
            missing_binaries.append(binary)

    if missing_binaries:
        error_message = (
            "The following HMMER tools are not found in your system PATH: "
            f"{', '.join(missing_binaries)}.\n\n"
            "Please install HMMER (version 3.1b2 or later) and ensure its tools are accessible."
            "You can download HMMER from http://hmmer.org/download.html\n\n"
            "Alternatively, if you are using Conda, you can install HMMER by running:\n"
            "  conda install -c bioconda hmmer\n\n"
            "Ensure that the directory containing the HMMER binaries is added to your PATH environment variable."
        )
        raise click.ClickException(error_message)

if __name__ == '__main__':
    # This is for testing the function directly
    try:
        check_hmmer_dependencies()
        print("All HMMER dependencies are satisfied.")
    except click.ClickException as e:
        print(e)
