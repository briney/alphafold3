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
import io
import sys
import importlib # Add this
import pathlib # Add this

# Make sure the module under test can be imported.
# This might require adjusting sys.path if tests are run from a different context.
# For now, assuming direct execution or pytest will handle paths.
from alphafold3.cli import health_check

class TestHealthCheck(unittest.TestCase):

    def setUp(self):
        # Redirect stdout to capture click.echo output for many tests
        self.mock_stdout = io.StringIO()
        # Some functions might use click.secho which can go to stderr by default if stdout is not a tty
        # or based on click's internal logic. For simplicity, we'll patch click.echo and click.secho
        # directly in tests where specific output capturing is needed.
        # For now, basic stdout redirection is a start.
        self.old_stdout = sys.stdout
        sys.stdout = self.mock_stdout

    def tearDown(self):
        sys.stdout = self.old_stdout # Restore stdout

    @mock.patch('shutil.which')
    def test_check_system_dependencies_all_found_verbose(self, mock_which):
        # Simulate all binaries are found
        mock_which.side_effect = lambda x: f"/usr/bin/{x}" # Return a path for any binary

        result = health_check.check_system_dependencies(verbose=True, fix=False)
        self.assertTrue(result)
        output = self.mock_stdout.getvalue()
        self.assertIn("Checking system dependencies...", output)
        self.assertIn("[OK] HMMER binaries found", output)
        self.assertIn("All system dependencies satisfied.", output)
        self.assertNotIn("[FAIL]", output)

    @mock.patch('shutil.which')
    def test_check_system_dependencies_all_found_non_verbose(self, mock_which):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        result = health_check.check_system_dependencies(verbose=False, fix=False)
        self.assertTrue(result)
        output = self.mock_stdout.getvalue()
        self.assertNotIn("Checking system dependencies...", output) # Verbose message
        self.assertNotIn("[OK] HMMER binaries found", output) # Verbose message
        self.assertNotIn("All system dependencies satisfied.", output) # Verbose message
        self.assertNotIn("[FAIL]", output) # No failures

    @mock.patch('shutil.which')
    def test_check_system_dependencies_one_missing(self, mock_which):
        # Simulate 'jackhmmer' is missing, others found
        def side_effect_func(binary_name):
            if binary_name == 'jackhmmer':
                return None
            return f"/usr/bin/{binary_name}"
        mock_which.side_effect = side_effect_func

        result = health_check.check_system_dependencies(verbose=False, fix=False)
        self.assertFalse(result)
        output = self.mock_stdout.getvalue()
        self.assertIn("[FAIL] Missing HMMER binaries: jackhmmer", output)
        self.assertIn("Suggestion: Install HMMER", output)
        self.assertIn("Some system dependencies are missing.", output)

    @mock.patch('shutil.which')
    def test_check_system_dependencies_one_missing_fix_mode(self, mock_which):
        def side_effect_func(binary_name):
            if binary_name == 'jackhmmer':
                return None
            return f"/usr/bin/{binary_name}"
        mock_which.side_effect = side_effect_func

        result = health_check.check_system_dependencies(verbose=False, fix=True)
        self.assertFalse(result)
        output = self.mock_stdout.getvalue()
        self.assertIn("[FAIL] Missing HMMER binaries: jackhmmer", output)
        self.assertIn("Suggestion: Install HMMER", output)
        self.assertIn("--fix mode: You could try running: `conda install -c bioconda hmmer`", output)

    @mock.patch('importlib.metadata.version')
    def test_check_python_packages_all_ok_verbose(self, mock_version):
        # Simulate all packages are installed and meet version requirements
        def version_side_effect(package_name):
            if package_name == "jax":
                return "0.4.34" # Meets requirement
            if package_name == "numpy":
                return "2.2.0" # Exceeds requirement "2.1.3"
            if package_name == "rdkit": # Example, actual required might be different
                return "2024.3.5"
            if package_name == "packaging": # No specific version, just presence
                return "23.0"
            # Add other packages from health_check.REQUIRED_PYTHON_PACKAGES as needed
            # For simplicity, mocking a few key ones.
            # A more robust test would iterate health_check.REQUIRED_PYTHON_PACKAGES
            # and provide mock versions for all.
            return "1.0.0" # Default for others
        mock_version.side_effect = version_side_effect

        result = health_check.check_python_packages(verbose=True, fix=False)
        self.assertTrue(result)
        output = self.mock_stdout.getvalue()
        self.assertIn("Checking Python package dependencies...", output)
        self.assertIn("[OK] jax version 0.4.34 (meets requirement 0.4.34)", output)
        self.assertIn("[OK] numpy version 2.2.0 (meets requirement 2.1.3)", output)
        self.assertIn("[OK] packaging version 23.0 (found)", output)
        self.assertIn("All Python package dependencies are satisfied.", output)
        self.assertNotIn("[FAIL]", output)

    @mock.patch('importlib.metadata.version')
    def test_check_python_packages_all_ok_non_verbose(self, mock_version):
        mock_version.return_value = "99.9.9" # Generic high version

        result = health_check.check_python_packages(verbose=False, fix=False)
        self.assertTrue(result)
        output = self.mock_stdout.getvalue()
        self.assertNotIn("Checking Python package dependencies...", output)
        self.assertNotIn("[OK] jax version", output) # Verbose message
        self.assertNotIn("All Python package dependencies are satisfied.", output) # Verbose message
        self.assertNotIn("[FAIL]", output)

    @mock.patch('importlib.metadata.version')
    def test_check_python_packages_one_missing(self, mock_version):
        def side_effect_func(package_name):
            if package_name == 'rdkit':
                raise importlib.metadata.PackageNotFoundError
            return "1.0.0" # Others found
        mock_version.side_effect = side_effect_func

        result = health_check.check_python_packages(verbose=False, fix=False)
        self.assertFalse(result)
        output = self.mock_stdout.getvalue()
        self.assertIn("[FAIL] rdkit is not installed.", output)
        self.assertIn("Suggestion: pip install \"rdkit==2024.3.5\"", output) # Check specific version suggestion
        self.assertIn("Some Python package dependencies are missing or outdated.", output)

    @mock.patch('importlib.metadata.version')
    def test_check_python_packages_one_outdated_fix_mode(self, mock_version):
        def side_effect_func(package_name):
            if package_name == 'jax':
                return "0.3.0" # Outdated, required is "0.4.34"
            return "99.9.9" # Others are fine
        mock_version.side_effect = side_effect_func

        result = health_check.check_python_packages(verbose=False, fix=True) # fix=True
        self.assertFalse(result)
        output = self.mock_stdout.getvalue()
        self.assertIn("[FAIL] jax is outdated (found 0.3.0, require 0.4.34)", output)
        self.assertIn("Suggestion: pip install --upgrade \"jax==0.4.34\"", output)
        self.assertIn("--fix mode: You could try running: `pip install --upgrade \"jax==0.4.34\"`", output)

    @mock.patch('importlib.metadata.version')
    def test_check_python_packages_no_version_requirement_found(self, mock_version):
        # Test for a package that only needs to be present (e.g., 'click' or 'packaging')
        mock_version.side_effect = lambda pkg: "1.2.3" if pkg == "click" else "0.1.0" # Mock other versions low

        result = health_check.check_python_packages(verbose=True, fix=False)
        # This test is a bit fragile if REQUIRED_PYTHON_PACKAGES changes.
        # We are mostly interested in the output for 'click'.
        # Assuming 'click' is in REQUIRED_PYTHON_PACKAGES with version None.
        # And other packages might fail version check to make result False, but we check click's output.
        output = self.mock_stdout.getvalue()
        self.assertIn("[OK] click version 1.2.3 (found)", output)

    @mock.patch('jax.local_devices')
    @mock.patch('jax.lib.xla_bridge.get_backend')
    def test_check_gpu_cuda_gpu_found_verbose(self, mock_get_backend, mock_local_devices):
        # Simulate one GPU found
        mock_gpu = mock.Mock()
        mock_gpu.platform = 'gpu'
        mock_gpu.device_kind = 'Test-GPU-Model-X'
        mock_gpu.id = 0
        mock_local_devices.return_value = [mock_gpu]

        # Simulate backend and platform version
        mock_backend_instance = mock.Mock()
        mock_backend_instance.platform_version = "CUDA 12.3"
        mock_get_backend.return_value = mock_backend_instance

        result = health_check.check_gpu_cuda(verbose=True)
        self.assertTrue(result)
        output = self.mock_stdout.getvalue()

        self.assertIn("Checking GPU/CUDA availability...", output)
        self.assertIn("Found device: gpu - Test-GPU-Model-X (id=0)", output)
        self.assertIn("[OK] Found 1 GPU(s) usable by JAX:", output)
        self.assertIn("GPU 0: Test-GPU-Model-X", output)
        self.assertIn("JAX platform version (includes CUDA info if applicable): CUDA 12.3", output)
        self.assertNotIn("[FAIL]", output)

    @mock.patch('jax.local_devices')
    def test_check_gpu_cuda_no_gpu_found_non_verbose(self, mock_local_devices):
        # Simulate only CPU found
        mock_cpu = mock.Mock()
        mock_cpu.platform = 'cpu'
        mock_cpu.device_kind = 'Test-CPU'
        mock_cpu.id = 0
        mock_local_devices.return_value = [mock_cpu]

        result = health_check.check_gpu_cuda(verbose=False)
        self.assertFalse(result)
        output = self.mock_stdout.getvalue()

        self.assertNotIn("Checking GPU/CUDA availability...", output) # Verbose
        self.assertIn("[FAIL] No GPUs found or JAX is not configured to use GPUs.", output)
        self.assertIn("Ensure NVIDIA drivers are correctly installed", output) # Suggestion

    @mock.patch('jax.local_devices', side_effect=Exception("JAX Error"))
    def test_check_gpu_cuda_jax_error(self, mock_local_devices_error):
        result = health_check.check_gpu_cuda(verbose=False)
        self.assertFalse(result)
        output = self.mock_stdout.getvalue()
        self.assertIn("[ERROR] An error occurred while checking for GPUs: JAX Error", output)

    @mock.patch('jax.local_devices')
    @mock.patch('jax.lib.xla_bridge.get_backend')
    def test_check_gpu_cuda_platform_version_error_verbose(self, mock_get_backend, mock_local_devices):
        # Simulate one GPU found
        mock_gpu = mock.Mock()
        mock_gpu.platform = 'gpu'
        mock_gpu.device_kind = 'Test-GPU-Model-Y'
        mock_gpu.id = 0
        mock_local_devices.return_value = [mock_gpu]

        # Simulate error getting platform version
        mock_get_backend.side_effect = Exception("Backend Error")

        result = health_check.check_gpu_cuda(verbose=True) # Verbose to check the error message
        self.assertTrue(result) # Still true because GPU was found
        output = self.mock_stdout.getvalue()

        self.assertIn("[OK] Found 1 GPU(s) usable by JAX:", output)
        self.assertIn("Could not determine JAX CUDA version details: Backend Error", output)

    @mock.patch('pathlib.Path.exists')
    @mock.patch('pathlib.Path.is_dir')
    @mock.patch('pathlib.Path.glob') # Or mock iterdir if that's what you used
    def test_check_model_database_availability_all_ok_verbose(self, mock_glob, mock_is_dir, mock_exists):
        # Simulate all files and directories exist and model dir is not empty
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_glob.return_value = [mock.Mock()] # Simulate model directory not empty

        result = health_check.check_model_database_availability(verbose=True)
        self.assertTrue(result)
        output = self.mock_stdout.getvalue()

        self.assertIn("Checking model and (miniature) database availability...", output)
        self.assertIn("[OK] Model directory found and appears to contain files: alphafold3_weights", output)
        self.assertIn("[OK] Found miniature database item: src/alphafold3/test_data/miniature_databases/uniref90__subsampled_1000.fasta", output)
        self.assertIn("[OK] Found miniature database item: src/alphafold3/test_data/miniature_databases/pdb_mmcif", output)
        self.assertIn("[OK] Found miniature database item: src/alphafold3/test_data/miniature_databases/pdb_mmcif/5y2e.cif", output)
        self.assertIn("Essential model and miniature database files appear to be available.", output)
        self.assertNotIn("[FAIL]", output)

    @mock.patch('pathlib.Path.exists')
    @mock.patch('pathlib.Path.is_dir')
    @mock.patch('pathlib.Path.glob')
    def test_check_model_database_availability_model_dir_missing(self, mock_glob, mock_is_dir, mock_exists):
        def exists_side_effect(path_obj):
            # Path objects are created like: pathlib.Path("alphafold3_weights")
            # So path_obj.name can be used to differentiate.
            if path_obj.name == "alphafold3_weights":
                return False # Model dir missing
            return True # Mini DBs are fine

        mock_exists.side_effect = exists_side_effect
        mock_is_dir.return_value = True # For mini_db_dir
        mock_glob.return_value = [mock.Mock()] # For mini_db_dir if it were checked for emptiness

        result = health_check.check_model_database_availability(verbose=False)
        self.assertFalse(result)
        output = self.mock_stdout.getvalue()

        self.assertIn("[FAIL] Model directory not found: alphafold3_weights", output)
        self.assertIn("Suggestion: Ensure models have been downloaded", output)
        self.assertIn("Some model or essential miniature database files are missing.", output)

    @mock.patch('pathlib.Path.exists')
    @mock.patch('pathlib.Path.is_dir')
    @mock.patch('pathlib.Path.glob')
    def test_check_model_database_availability_model_dir_empty_verbose(self, mock_glob, mock_is_dir, mock_exists):
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        # Simulate model directory is empty
        def glob_side_effect(pattern): # pattern is '*'
            # We need to differentiate which Path object glob is called on.
            # This is tricky if Path itself is not mocked more extensively.
            # Assuming the first call to glob is for MODEL_DIR
            if not hasattr(glob_side_effect, 'called_once'):
                glob_side_effect.called_once = True
                return [] # Empty for model_dir
            return [mock.Mock()] # Not empty for other potential glob calls (none in current func)
        mock_glob.side_effect = glob_side_effect


        result = health_check.check_model_database_availability(verbose=True)
        self.assertFalse(result)
        output = self.mock_stdout.getvalue()

        self.assertIn("[FAIL] Model directory found but is empty: alphafold3_weights", output)
        self.assertIn("Suggestion: Ensure models have been downloaded", output)
        # Check that mini dbs are still reported as OK
        self.assertIn("[OK] Found miniature database item: src/alphafold3/test_data/miniature_databases/uniref90__subsampled_1000.fasta", output)
        self.assertIn("Some model or essential miniature database files are missing.", output)


    @mock.patch('pathlib.Path.exists')
    @mock.patch('pathlib.Path.is_dir')
    def test_check_model_database_availability_mini_db_file_missing(self, mock_is_dir, mock_exists):
        def exists_side_effect(path_obj):
            # path_obj is a pathlib.Path object.
            # Check for a specific mini_db file missing
            if "uniref90__subsampled_1000.fasta" in str(path_obj):
                return False
            return True # All other files/dirs exist

        mock_exists.side_effect = exists_side_effect
        mock_is_dir.return_value = True # All relevant dirs exist
        # Need to mock glob for the model dir check to pass
        with mock.patch('pathlib.Path.glob', return_value=[mock.Mock()]):
            result = health_check.check_model_database_availability(verbose=False)

        self.assertFalse(result)
        output = self.mock_stdout.getvalue()

        self.assertIn("[FAIL] Missing miniature database item: src/alphafold3/test_data/miniature_databases/uniref90__subsampled_1000.fasta", output)
        self.assertIn("Suggestion: This item should be part of the AlphaFold 3 source code test_data.", output)
        self.assertIn("Some model or essential miniature database files are missing.", output)

    # For run_quick_functionality_test, mocking will be more involved.
    # We need to mock external calls and classes it uses.
    # Patching locations should be where they are looked up (i.e., in health_check module)

    @mock.patch('alphafold3.cli.health_check.pathlib.Path')
    @mock.patch('alphafold3.cli.health_check.shutil.which')
    @mock.patch('alphafold3.cli.health_check.jax')
    @mock.patch('alphafold3.cli.health_check.af3_pipeline.DataPipeline') # Note: af3_pipeline alias
    @mock.patch('alphafold3.cli.health_check.featurisation.featurise_input')
    @mock.patch('alphafold3.cli.health_check.run_alphafold.ModelRunner')
    @mock.patch('alphafold3.cli.health_check.chemical_components.cached_ccd')
    def test_run_quick_functionality_test_success_verbose(
        self, mock_cached_ccd, mock_model_runner, mock_featurise_input,
        mock_data_pipeline, mock_jax, mock_which, mock_pathlib_path):

        # --- Setup Mocks ---
        # 1. Path checks (MODEL_DIR, MINI_DB_ROOT)
        mock_model_dir_instance = mock.Mock()
        mock_model_dir_instance.exists.return_value = True
        mock_model_dir_instance.is_dir.return_value = True
        mock_model_dir_instance.glob.return_value = [mock.Mock()] # Not empty

        mock_mini_db_root_instance = mock.Mock()
        mock_mini_db_root_instance.exists.return_value = True

        # pathlib.Path() is called for MODEL_DIR and for MINI_DB_ROOT (via __file__)
        # We need side_effect to return different mocks based on input.
        def path_side_effect(path_arg):
            if str(path_arg) == "alphafold3_weights":
                return mock_model_dir_instance
            # This part is tricky because MINI_DB_ROOT is derived from __file__
            # For the unit test, let's assume the path construction for MINI_DB_ROOT works
            # and just mock its .exists() outcome.
            # A simpler way: mock the final MINI_DB_ROOT.exists() call if possible,
            # or ensure the mock_pathlib_path covers the specific path string.
            # The current mock_pathlib_path is for the class, so its instances are auto-mocks.
            # We'll rely on the instance mock for MINI_DB_ROOT.exists.
            # Let's make it simple: all Path("...").exists() calls return True from this mock_pathlib_path
            # and specific instances above handle specific paths.
            # This part of path mocking can be fragile.
            # A better approach for MINI_DB_ROOT might be to mock __file__ if it were simpler.
            # For now, let's assume MINI_DB_ROOT.exists() will be true.
            # The provided code for health_check.py uses `pathlib.Path(__file__).parent.parent / "test_data" / "miniature_databases"`
            # We can make the main mock_pathlib_path return a general mock that has exists=True
            mock_path_instance = mock.Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_dir.return_value = True # For MINI_DB_ROOT check
            mock_path_instance.resolve.return_value = mock_path_instance # for MINI_DB_ROOT.resolve()

            # Specific mocks for MODEL_DIR
            if str(path_arg) == "alphafold3_weights":
                 return mock_model_dir_instance
            # For MINI_DB_ROOT, the path is more complex.
            # We will assume the path is constructed and then .exists() is called.
            # The mock_pathlib_path.return_value.exists.return_value = True should cover it.
            return mock_path_instance

        mock_pathlib_path.side_effect = path_side_effect
        # This is still a bit broad. Let's refine the path mocking if issues arise.
        # A common strategy is to mock specific Path instances if they are stored,
        # or mock the methods on the Path class prototype.

        # 2. shutil.which (HMMER tools)
        mock_which.return_value = "/fake/bin/tool"

        # 3. JAX devices
        mock_gpu_device = mock.Mock()
        mock_gpu_device.platform = 'gpu'
        mock_gpu_device.device_kind = 'Mocked-GPU'
        mock_jax.local_devices.return_value = [mock_gpu_device] # Ensure it returns a list

        # 4. DataPipeline
        mock_data_pipeline_instance = mock_data_pipeline.return_value
        mock_data_pipeline_instance.process.return_value = mock.Mock() # Processed input

        # 5. Featurise input
        mock_featurised_example = mock.Mock()
        mock_featurise_input.return_value = [mock_featurised_example] # List of examples

        # 6. ModelRunner
        mock_model_runner_instance = mock_model_runner.return_value
        mock_inference_result = {'predicted_structure': mock.Mock(coords=[1,2,3])}
        mock_model_runner_instance.run_inference.return_value = mock_inference_result

        # 7. cached_ccd
        mock_cached_ccd.return_value = mock.Mock()


        # --- Run Test ---
        result = health_check.run_quick_functionality_test(verbose=True)
        self.assertTrue(result)
        output = self.mock_stdout.getvalue()

        self.assertIn("Running quick functionality test", output)
        self.assertIn("Using JAX device: gpu - Mocked-GPU", output)
        self.assertIn("Initializing DataPipeline...", output)
        self.assertIn("Processing input: health_check_test_peptide...", output)
        self.assertIn("Featurizing input...", output)
        self.assertIn("Initializing ModelRunner...", output)
        self.assertIn("Running inference (on minimal input)...", output)
        self.assertIn("[OK] Quick functionality test completed successfully.", output)
        self.assertNotIn("[FAIL]", output)
        self.assertNotIn("[SKIP]", output)

    @mock.patch('alphafold3.cli.health_check.pathlib.Path')
    def test_run_quick_functionality_test_model_dir_empty(self, mock_pathlib_path):
        mock_model_dir_instance = mock.Mock()
        mock_model_dir_instance.exists.return_value = True
        mock_model_dir_instance.is_dir.return_value = True
        mock_model_dir_instance.glob.return_value = [] # Empty model dir

        # Only mock the relevant Path call
        mock_pathlib_path.side_effect = lambda p: mock_model_dir_instance if str(p) == "alphafold3_weights" else mock.DEFAULT

        result = health_check.run_quick_functionality_test(verbose=False)
        self.assertFalse(result) # Should fail/skip if models are missing
        output = self.mock_stdout.getvalue()
        self.assertIn("[SKIP] Quick functionality test: Model directory is missing or empty.", output)

    @mock.patch('alphafold3.cli.health_check.pathlib.Path', mock.Mock(exists=lambda:True, is_dir=lambda:True, glob=lambda p: [mock.Mock()])) # Model dir OK
    @mock.patch('alphafold3.cli.health_check.shutil.which', return_value=None) # HMMER missing
    def test_run_quick_functionality_test_hmmer_missing(self, mock_which):
        # Need to also mock JAX to avoid it erroring out if not configured
        with mock.patch('alphafold3.cli.health_check.jax.local_devices', return_value=[mock.Mock(platform='cpu')]):
            result = health_check.run_quick_functionality_test(verbose=False)
        self.assertFalse(result)
        output = self.mock_stdout.getvalue()
        self.assertIn("[FAIL] Quick functionality test: Missing HMMER binaries", output)

    @mock.patch('alphafold3.cli.health_check.pathlib.Path')
    @mock.patch('alphafold3.cli.health_check.shutil.which', return_value="/fake/tool")
    @mock.patch('alphafold3.cli.health_check.jax.local_devices', return_value=[mock.Mock(platform='gpu')])
    @mock.patch('alphafold3.cli.health_check.af3_pipeline.DataPipeline.process', side_effect=Exception("Pipeline Error"))
    # Mock other necessary parts for the setup to pass up to the point of failure
    @mock.patch('alphafold3.cli.health_check.chemical_components.cached_ccd', return_value=mock.Mock())
    def test_run_quick_functionality_test_pipeline_error_verbose(
        self, mock_ccd, mock_pipeline_process, mock_path_constructor):

        # Setup Path mocks for MODEL_DIR and MINI_DB_ROOT to exist and be valid
        mock_model_dir_instance = mock.Mock()
        mock_model_dir_instance.exists.return_value = True
        mock_model_dir_instance.is_dir.return_value = True
        mock_model_dir_instance.glob.return_value = [mock.Mock()]

        # Mock for MINI_DB_ROOT and its parent structure
        # This path mocking is the most complex part
        mock_file_path = mock.Mock() # Mocks __file__
        mock_parent1 = mock.Mock()   # Mocks __file__.parent
        mock_parent2 = mock.Mock()   # Mocks __file__.parent.parent
        mock_test_data = mock.Mock() # Mocks .../test_data
        mock_mini_db = mock.Mock()   # Mocks .../miniature_databases

        mock_mini_db.exists.return_value = True
        mock_mini_db.is_dir.return_value = True
        mock_mini_db.resolve.return_value = mock_mini_db

        mock_test_data.__truediv__.return_value = mock_mini_db
        mock_parent2.__truediv__.return_value = mock_test_data
        mock_parent1.parent = mock_parent2
        mock_file_path.parent = mock_parent1

        def path_side_effect(p_arg):
            if str(p_arg) == "alphafold3_weights":
                return mock_model_dir_instance
            elif p_arg == pathlib.Path(__file__): # This won't work directly as __file__ is of the test
                                                 # We need to mock how health_check.pathlib.Path(__file__) behaves
                                                 # This should be health_check.pathlib.Path(health_check.__file__)
                return mock_file_path
            # Fallback for other Path calls if any
            m = mock.Mock()
            m.exists.return_value = True
            m.is_dir.return_value = True
            m.glob.return_value = [mock.Mock()]
            m.resolve.return_value = m
            return m

        # The __file__ in health_check.py is what we need to control
        # So, we patch health_check.pathlib.Path
        # The instance of Path created with __file__ inside health_check.py needs to be controlled.
        # This is still a bit tricky. Let's assume the path resolution for MINI_DB_ROOT works
        # and its .exists() is true. The mock_path_constructor above can be simplified for this test
        # to just ensure all .exists() calls return True for paths.

        # Simplified path mock for this specific test:
        mock_path_instance = mock.Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_dir.return_value = True
        mock_path_instance.glob.return_value = [mock.Mock()]
        mock_path_instance.resolve.return_value = mock_path_instance # for MINI_DB_ROOT.resolve()
        mock_path_constructor.return_value = mock_path_instance # All Path() calls return this.

        result = health_check.run_quick_functionality_test(verbose=True)
        self.assertFalse(result)
        output = self.mock_stdout.getvalue()
        self.assertIn("[FAIL] Quick functionality test encountered an error: Pipeline Error", output)
        self.assertIn("Traceback:", output) # Check for traceback in verbose

if __name__ == '__main__':
    unittest.main()
