import unittest
from unittest import mock
import os
import json
from pathlib import Path
import tempfile
import shutil

# Attempt to import the functions to be tested
# This structure assumes 'src' is in PYTHONPATH or tests are run in a way that resolves this.
try:
    from alphafold3.config import paths
except ImportError:
    # Fallback for environments where 'src' is not directly in path
    # This might require adjustments based on the specific test runner setup
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])) # Add 'src' to path
    from alphafold3.config import paths

class TestAlphaFold3Paths(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for fake home, config files, models, dbs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fake_home = Path(self.temp_dir.name) / "fake_home"
        self.fake_home.mkdir()

        # Fake directories for default locations
        self.default_model_dir_user = self.fake_home / ".alphafold3" / "models"
        self.default_db_dir_user = self.fake_home / ".alphafold3" / "databases"
        self.default_model_dir_user.mkdir(parents=True, exist_ok=True)
        self.default_db_dir_user.mkdir(parents=True, exist_ok=True)

        # Paths for fake config files
        self.fake_config_dir1 = self.fake_home / ".alphafold3"
        self.fake_config_dir2 = self.fake_home / ".config" / "alphafold3"
        self.fake_config_dir1.mkdir(parents=True, exist_ok=True)
        self.fake_config_dir2.mkdir(parents=True, exist_ok=True)

        self.config_file_path1 = self.fake_config_dir1 / "config.json"
        self.config_file_path2 = self.fake_config_dir2 / "config.json"

        # Mock Path.home()
        self.patch_home = mock.patch('pathlib.Path.home', return_value=self.fake_home)
        self.mock_home = self.patch_home.start()

        # Mock os.access to assume readability unless specifically denied for a test
        self.patch_os_access = mock.patch('os.access', return_value=True)
        self.mock_os_access = self.patch_os_access.start()

        # Reset any cached config in the paths module before each test
        paths._config_from_file = {} # Clear cached config
        # paths._load_config_from_file() # Re-evaluate with mocked home, or mock directly

    def tearDown(self):
        self.patch_home.stop()
        self.patch_os_access.stop()
        self.temp_dir.cleanup()
        paths._config_from_file = {} # Clean up global state post-test

    def _create_config_file(self, content, path_choice=1):
        config_path = self.config_file_path1 if path_choice == 1 else self.config_file_path2
        with open(config_path, 'w') as f:
            json.dump(content, f)
        # Reload the config in the paths module as it caches it at import time
        # We need to mock paths.CONFIG_FILE_LOCATIONS for the duration of _load_config_from_file
        # or directly mock _load_config_from_file's return value.
        # For simplicity, let's assume we can directly influence paths._config_from_file for tests.
        paths._config_from_file = content


    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_model_dir_cli_arg(self):
        # Test CLI argument has highest priority
        cli_path = self.fake_home / "cli_models"
        cli_path.mkdir()
        # Override default search paths for this test to ensure CLI is picked
        with mock.patch('alphafold3.config.paths.DEFAULT_MODEL_SEARCH_DIRS', []):
             resolved_path = paths.get_model_dir(cli_model_dir=str(cli_path))
        self.assertEqual(resolved_path, cli_path.resolve())

    @mock.patch.dict(os.environ, {'AF3_MODEL_DIR': str(Path(tempfile.gettempdir()) / 'env_models')}, clear=True)
    def test_get_model_dir_env_var(self):
        # Test environment variable priority
        env_path_str = os.environ['AF3_MODEL_DIR']
        env_path = Path(env_path_str)
        env_path.mkdir(exist_ok=True) # Create it
        # Override default search paths for this test
        with mock.patch('alphafold3.config.paths.DEFAULT_MODEL_SEARCH_DIRS', []):
            resolved_path = paths.get_model_dir(cli_model_dir=None)
        self.assertEqual(resolved_path, env_path.resolve())
        shutil.rmtree(env_path, ignore_errors=True) # Clean up

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_model_dir_config_file(self):
        # Test config file priority
        config_model_path = self.fake_home / "config_models"
        config_model_path.mkdir()
        self._create_config_file({"model_dir": str(config_model_path)})

        # Mock DEFAULT_MODEL_SEARCH_DIRS to be empty to ensure config is used
        with mock.patch('alphafold3.config.paths.DEFAULT_MODEL_SEARCH_DIRS', []):
             # Need to re-trigger load or mock what _load_config_from_file returns
            with mock.patch('alphafold3.config.paths._load_config_from_file', return_value={"model_dir": str(config_model_path)}):
                resolved_path = paths.get_model_dir(cli_model_dir=None)
        self.assertEqual(resolved_path, config_model_path.resolve())

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_model_dir_default_location(self):
        # Test default location
        # self.default_model_dir_user is already created in setUp
        # Ensure no CLI, ENV, or config file is set
        paths._config_from_file = {} # Ensure no config from file
        with mock.patch('alphafold3.config.paths.DEFAULT_MODEL_SEARCH_DIRS', [self.default_model_dir_user]):
             resolved_path = paths.get_model_dir(cli_model_dir=None)
        self.assertEqual(resolved_path, self.default_model_dir_user.resolve())

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_model_dir_not_found(self):
        # Test FileNotFoundError when no path is valid
        paths._config_from_file = {}
        with mock.patch('alphafold3.config.paths.DEFAULT_MODEL_SEARCH_DIRS', [self.fake_home / "non_existent_default"]):
            with self.assertRaises(FileNotFoundError):
                paths.get_model_dir(cli_model_dir=None)

    # --- Tests for get_database_dirs ---
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_database_dirs_cli_arg(self):
        cli_db_path1 = self.fake_home / "cli_db1"
        cli_db_path1.mkdir()
        cli_db_path2 = self.fake_home / "cli_db2"
        cli_db_path2.mkdir()
        with mock.patch('alphafold3.config.paths.DEFAULT_DB_SEARCH_DIRS', []):
            resolved_paths = paths.get_database_dirs(cli_db_dirs=[str(cli_db_path1), str(cli_db_path2)])
        self.assertListEqual(sorted(resolved_paths), sorted([cli_db_path1.resolve(), cli_db_path2.resolve()]))

    @mock.patch.dict(os.environ, {'AF3_DB_DIRS': tempfile.gettempdir() + '/env_db1,' + tempfile.gettempdir() + '/env_db2'}, clear=True)
    def test_get_database_dirs_env_var(self):
        env_db_path1_str = os.environ['AF3_DB_DIRS'].split(',')[0]
        env_db_path2_str = os.environ['AF3_DB_DIRS'].split(',')[1]
        env_db_path1 = Path(env_db_path1_str)
        env_db_path2 = Path(env_db_path2_str)
        env_db_path1.mkdir(exist_ok=True)
        env_db_path2.mkdir(exist_ok=True)

        with mock.patch('alphafold3.config.paths.DEFAULT_DB_SEARCH_DIRS', []):
            resolved_paths = paths.get_database_dirs(cli_db_dirs=None)
        self.assertListEqual(sorted(resolved_paths), sorted([env_db_path1.resolve(), env_db_path2.resolve()]))
        shutil.rmtree(env_db_path1, ignore_errors=True)
        shutil.rmtree(env_db_path2, ignore_errors=True)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_database_dirs_config_file(self):
        config_db_path1 = self.fake_home / "config_db1"
        config_db_path1.mkdir()
        config_db_path2 = self.fake_home / "config_db2"
        config_db_path2.mkdir()
        self._create_config_file({"database_dirs": [str(config_db_path1), str(config_db_path2)]})

        with mock.patch('alphafold3.config.paths.DEFAULT_DB_SEARCH_DIRS', []):
            with mock.patch('alphafold3.config.paths._load_config_from_file', return_value={"database_dirs": [str(config_db_path1), str(config_db_path2)]}):
                resolved_paths = paths.get_database_dirs(cli_db_dirs=None)
        self.assertListEqual(sorted(resolved_paths), sorted([config_db_path1.resolve(), config_db_path2.resolve()]))

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_database_dirs_default_location(self):
        # self.default_db_dir_user is created in setUp
        paths._config_from_file = {}
        with mock.patch('alphafold3.config.paths.DEFAULT_DB_SEARCH_DIRS', [self.default_db_dir_user]):
             resolved_paths = paths.get_database_dirs(cli_db_dirs=None)
        self.assertListEqual(resolved_paths, [self.default_db_dir_user.resolve()])

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_database_dirs_not_found(self):
        paths._config_from_file = {}
        with mock.patch('alphafold3.config.paths.DEFAULT_DB_SEARCH_DIRS', [self.fake_home / "non_existent_db_default"]):
            with self.assertRaises(FileNotFoundError):
                paths.get_database_dirs(cli_db_dirs=None)

    # --- Tests for find_database_file ---
    def test_find_database_file_success_relative(self):
        db_dir = self.default_db_dir_user # Exists
        (db_dir / "mgnify").mkdir()
        target_file = db_dir / "mgnify" / "mgy_clusters_2022_05.fa"
        target_file.touch()

        found_path = paths.find_database_file([db_dir], "mgnify/mgy_clusters_2022_05.fa")
        self.assertEqual(found_path, target_file.resolve())

    def test_find_database_file_success_db_dir_placeholder(self):
        db_dir = self.default_db_dir_user
        (db_dir / "bfd").mkdir(exist_ok=True) # ensure bfd subdir exists
        target_file = db_dir / "bfd" / "bfd_file.fasta"
        target_file.touch()

        found_path = paths.find_database_file([db_dir], "${DB_DIR}/bfd/bfd_file.fasta")
        self.assertEqual(found_path, target_file.resolve())

    def test_find_database_file_not_found(self):
        db_dir = self.default_db_dir_user
        with self.assertRaises(FileNotFoundError):
            paths.find_database_file([db_dir], "non_existent_file.fasta")

    def test_find_database_file_multiple_dirs(self):
        db_dir1 = self.default_db_dir_user
        db_dir2 = self.fake_home / "other_db_dir"
        db_dir2.mkdir()
        (db_dir2 / "mgnify").mkdir()
        target_file = db_dir2 / "mgnify" / "mgy_clusters.fa"
        target_file.touch()

        # File not in db_dir1, but in db_dir2
        found_path = paths.find_database_file([db_dir1, db_dir2], "mgnify/mgy_clusters.fa")
        self.assertEqual(found_path, target_file.resolve())


if __name__ == '__main__':
    unittest.main()
