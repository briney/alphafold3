# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

import unittest
from unittest import mock
import click

from alphafold3.dependencies.system_deps import check_hmmer_dependencies, HMMER_BINARIES

class TestSystemDeps(unittest.TestCase):

    @mock.patch('shutil.which')
    def test_all_dependencies_present(self, mock_which):
        """Tests that no exception is raised when all HMMER binaries are found."""
        mock_which.return_value = '/usr/bin/some_binary'  # Simulate binary found
        try:
            check_hmmer_dependencies()
        except click.ClickException:
            self.fail("check_hmmer_dependencies() raised ClickException unexpectedly!")

    @mock.patch('shutil.which')
    def test_one_dependency_missing(self, mock_which):
        """Tests that ClickException is raised when one HMMER binary is missing."""
        # Simulate one binary missing, others present
        missing_binary = HMMER_BINARIES[0]
        def side_effect(binary_name):
            if binary_name == missing_binary:
                return None
            return f'/usr/bin/{binary_name}'
        mock_which.side_effect = side_effect

        with self.assertRaises(click.ClickException) as context:
            check_hmmer_dependencies()

        self.assertIn(f"The following HMMER tools are not found in your system PATH: {missing_binary}", str(context.exception))
        self.assertIn("Please install HMMER", str(context.exception))

    @mock.patch('shutil.which')
    def test_all_dependencies_missing(self, mock_which):
        """Tests that ClickException is raised when all HMMER binaries are missing."""
        mock_which.return_value = None  # Simulate all binaries missing

        with self.assertRaises(click.ClickException) as context:
            check_hmmer_dependencies()

        expected_missing_list = ", ".join(HMMER_BINARIES)
        self.assertIn(f"The following HMMER tools are not found in your system PATH: {expected_missing_list}", str(context.exception))
        self.assertIn("Please install HMMER", str(context.exception))

    @mock.patch('shutil.which')
    def test_some_dependencies_missing(self, mock_which):
        """Tests that ClickException is raised with a correct list of multiple missing HMMER binaries."""
        missing_binaries = [HMMER_BINARIES[0], HMMER_BINARIES[2]]
        present_binaries = [b for b in HMMER_BINARIES if b not in missing_binaries]

        def side_effect(binary_name):
            if binary_name in missing_binaries:
                return None
            return f'/usr/bin/{binary_name}'
        mock_which.side_effect = side_effect

        with self.assertRaises(click.ClickException) as context:
            check_hmmer_dependencies()

        expected_missing_list = ", ".join(missing_binaries)
        self.assertIn(f"The following HMMER tools are not found in your system PATH: {expected_missing_list}", str(context.exception))
        self.assertIn("Please install HMMER", str(context.exception))
        # Check that present binaries are not listed as missing
        for pb in present_binaries:
            # Extract just the list of missing binaries from the exception message for this check
            # The message format is "The following HMMER tools are not found in your system PATH: BIN_A, BIN_B.\n\n..."
            full_message = str(context.exception)
            # Get the part after "PATH: " and before ".\n\n"
            try:
                reported_missing_str = full_message.split("PATH: ")[1].split(".\n\n")[0]
                self.assertNotIn(pb, reported_missing_str.split(", "))
            except IndexError:
                self.fail(f"Could not parse missing binaries from exception message: {full_message}")


if __name__ == '__main__':
    unittest.main()
