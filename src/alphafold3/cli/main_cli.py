# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Main CLI group for AlphaFold 3."""

import click
from alphafold3.dependencies.system_deps import check_hmmer_dependencies

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def alphafold3():
    """AlphaFold 3 command line tools."""
    try:
        check_hmmer_dependencies()
    except click.ClickException as e:
        # click.echo(str(e), err=True) # This would print the message twice
        raise e # Re-raise the exception to let Click handle the exit and message display

if __name__ == '__main__':
    alphafold3()
