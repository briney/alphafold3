# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""CLI entry point for running AlphaFold 3 tests."""

import click
import subprocess
import sys
import pathlib

@click.command(context_settings=dict(
    ignore_unknown_options=True,  # Allows passing arbitrary flags to the script
    allow_extra_args=True,        # Collects all extra arguments
))
@click.argument('test_args', nargs=-1, type=click.UNPROCESSED)
def main(test_args):
    """Runs the AlphaFold 3 test suite (run_alphafold_test.py).

    Any additional arguments are passed directly to run_alphafold_test.py.
    """
    # Determine the project root. This assumes the script is located at:
    # <project_root>/src/alphafold3/cli/run_alphafold_test.py
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    test_script_path = project_root / "run_alphafold_test.py"

    if not test_script_path.is_file():
        click.echo(
            f"Error: Test script {test_script_path} not found. "
            "Make sure it exists at the project root.",
            err=True
        )
        sys.exit(1)

    command = [sys.executable, str(test_script_path)] + list(test_args)

    # Echo the command being run, helpful for debugging
    # Use shlex.join for safer and more accurate command string representation if complex arguments were possible,
    # but for simple list joining, this is fine.
    click.echo(f"Executing: {' '.join(command)}", err=True) # err=True to send to stderr

    try:
        # Using Popen to stream output in real-time.
        # stdout and stderr are directly connected to the parent's streams.
        process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)

        # Wait for the process to complete.
        process.wait()

        # If the process exited with a non-zero code, this script should also exit with that code.
        if process.returncode != 0:
            # Error message is not strictly needed here as run_alphafold_test.py's output will be visible.
            # click.echo(f"Test script exited with error code {process.returncode}.", err=True)
            sys.exit(process.returncode)

    except FileNotFoundError:
        # This error occurs if sys.executable or test_script_path is not found by Popen.
        click.echo(
            f"Error: Python interpreter '{sys.executable}' or test script '{test_script_path}' not found.",
            err=True
        )
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected exceptions during Popen or wait.
        click.echo(f"An unexpected error occurred while trying to run the test script: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
