import os
import json
from pathlib import Path
from typing import List, Optional, Union

# Define default search paths
# These will be used if no other configuration is found.
# Order matters: the first existing and readable path will be preferred.
DEFAULT_MODEL_SEARCH_DIRS: List[Path] = [
    Path.home() / '.alphafold3' / 'models',
    Path('/opt/alphafold3/models'),
    Path('~/alphafold3/models').expanduser(), # For user convenience
]

DEFAULT_DB_SEARCH_DIRS: List[Path] = [
    Path.home() / '.alphafold3' / 'databases',
    Path('/opt/alphafold3/databases'),
    Path('~/alphafold3/databases').expanduser(), # For user convenience
]

# Potential config file locations
CONFIG_FILE_LOCATIONS: List[Path] = [
    Path.home() / '.alphafold3' / 'config.json',
    Path.home() / '.config' / 'alphafold3' / 'config.json',
]

def _load_config_from_file() -> dict:
    """Loads configuration from a JSON file if it exists."""
    for config_path in CONFIG_FILE_LOCATIONS:
        if config_path.exists() and config_path.is_file():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")
    return {}

_config_from_file = _load_config_from_file()

def _find_first_valid_path(paths: List[Union[str, Path]]) -> Optional[Path]:
    """Finds the first existing and readable directory from a list of paths."""
    for p_str in paths:
        if not p_str:  # Skip empty or None paths
            continue
        p = Path(p_str).expanduser().resolve()
        if p.exists() and p.is_dir() and os.access(p, os.R_OK):
            return p
    return None

def get_model_dir(cli_model_dir: Optional[str] = None) -> Path:
    """
    Resolves the model directory based on the configuration hierarchy.
    Hierarchy:
    1. Command-line argument (`cli_model_dir`)
    2. Environment variable (`AF3_MODEL_DIR`)
    3. Configuration file (`~/.alphafold3/config.json` or `~/.config/alphafold3/config.json`)
    4. Default search locations.
    """
    paths_to_check: List[Optional[Union[str, Path]]] = []

    # 1. Command-line argument
    if cli_model_dir:
        paths_to_check.append(cli_model_dir)

    # 2. Environment variable
    env_model_dir = os.environ.get('AF3_MODEL_DIR')
    if env_model_dir:
        paths_to_check.append(env_model_dir)

    # 3. Configuration file
    file_config_model_dir = _config_from_file.get('model_dir')
    if file_config_model_dir:
        paths_to_check.append(file_config_model_dir)

    # Add defaults to the end of the list to check
    paths_to_check.extend(DEFAULT_MODEL_SEARCH_DIRS)

    resolved_path = _find_first_valid_path(paths_to_check)
    if resolved_path:
        return resolved_path

    # If no path is found after checking all sources including defaults
    raise FileNotFoundError(
        "Model directory not found. Please specify it via command-line (--model_dir), "
        "environment variable (AF3_MODEL_DIR), a configuration file, "
        f"or ensure it exists in one of the default locations: {DEFAULT_MODEL_SEARCH_DIRS}."
    )

def get_database_dirs(cli_db_dirs: Optional[List[str]] = None) -> List[Path]:
    """
    Resolves database directories based on the configuration hierarchy.
    Hierarchy:
    1. Command-line argument (`cli_db_dirs`)
    2. Environment variable (`AF3_DB_DIRS` - comma-separated)
    3. Configuration file (`database_dirs` - a list of paths)
    4. Default search locations.

    Returns a list of valid database directories.
    If a configured path is not found, it's ignored. If no paths are found
    after checking all sources, raises FileNotFoundError.
    """
    candidate_paths: List[Union[str, Path]] = []

    # 1. Command-line arguments
    if cli_db_dirs:
        candidate_paths.extend(cli_db_dirs)

    # 2. Environment variable (comma-separated list)
    env_db_dirs_str = os.environ.get('AF3_DB_DIRS')
    if env_db_dirs_str:
        candidate_paths.extend([p.strip() for p in env_db_dirs_str.split(',')])

    # 3. Configuration file (list of paths)
    file_config_db_dirs = _config_from_file.get('database_dirs')
    if isinstance(file_config_db_dirs, list):
        candidate_paths.extend(file_config_db_dirs)
    elif file_config_db_dirs: # if it's a single string path
        candidate_paths.append(file_config_db_dirs)

    # 4. Default search locations
    # We add these as individual candidates to be validated.
    # _find_first_valid_path is for a single path, here we want all valid default paths if specified by user
    # or if primary methods fail.

    resolved_paths: List[Path] = []
    checked_paths_for_error_msg = set()

    # First, try to resolve paths from CLI, Env, Config
    for p_str in candidate_paths:
        if not p_str: continue
        p = Path(p_str).expanduser().resolve()
        checked_paths_for_error_msg.add(str(p))
        if p.exists() and p.is_dir() and os.access(p, os.R_OK):
            if p not in resolved_paths: # Avoid duplicates
                 resolved_paths.append(p)

    # If any paths were resolved from CLI, Env, or Config, return them.
    if resolved_paths:
        return resolved_paths

    # If no paths from CLI, Env, Config, then check default locations.
    # We want to return *all* valid default locations if no specific user config was found and valid.
    for default_dir in DEFAULT_DB_SEARCH_DIRS:
        p = default_dir.expanduser().resolve() # Already Path objects
        checked_paths_for_error_msg.add(str(p))
        if p.exists() and p.is_dir() and os.access(p, os.R_OK):
            if p not in resolved_paths: # Avoid duplicates
                resolved_paths.append(p)

    if resolved_paths:
        return resolved_paths

    raise FileNotFoundError(
        "No valid database directories found. Please specify them via command-line (--db-dir), "
        "environment variable (AF3_DB_DIRS), a configuration file, "
        f"or ensure they exist in one of the default locations: {DEFAULT_DB_SEARCH_DIRS}. "
        f"Checked: {', '.join(sorted(list(checked_paths_for_error_msg)))}"
    )

def find_database_file(base_dirs: List[Path], file_path_template: str) -> Path:
    """
    Searches for a database file or directory within a list of base directories.
    The file_path_template can contain ${DB_DIR} which will be replaced by each base_dir.
    If ${DB_DIR} is not in the template, it's treated as a relative path from base_dirs.

    Args:
        base_dirs: A list of parent directories to search within.
        file_path_template: The path template for the file/directory to find.
                            Can be a specific name (e.g., "mgnify/mgy_clusters_2022_05.fa")
                            or include ${DB_DIR} (e.g., "${DB_DIR}/bfd/bfd.mgnify.fasta").

    Returns:
        The Path to the first found file/directory.

    Raises:
        FileNotFoundError: If the file/directory is not found in any of the base_dirs.
    """

    for db_dir in base_dirs:
        db_dir = Path(db_dir).expanduser().resolve()
        if "${DB_DIR}" in file_path_template:
            # Replace placeholder and check
            # Ensure that db_dir is treated as a string for substitution
            path_to_check_str = file_path_template.replace("${DB_DIR}", str(db_dir))
            path_to_check = Path(path_to_check_str)
        else:
            # Treat as relative path from the db_dir
            path_to_check = db_dir / file_path_template

        if path_to_check.exists(): # Could be file or directory
            return path_to_check

    raise FileNotFoundError(
        f"Database file/directory '{file_path_template}' not found in any of the "
        f"provided database directories: {[str(d) for d in base_dirs]}."
    )

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print("Testing configuration loading...")

    # Mock CLI arguments
    mock_cli_model_dir = None
    mock_cli_db_dirs = None

    # To test environment variables, you would set them before running this script, e.g.:
    # export AF3_MODEL_DIR=/tmp/my_models
    # export AF3_DB_DIRS=/tmp/my_dbs1,/tmp/my_dbs2

    # To test config file, create a ~/.alphafold3/config.json or ~/.config/alphafold3/config.json
    # Example: {"model_dir": "/tmp/config_models", "database_dirs": ["/tmp/config_dbs"]}

    # Create some dummy directories for testing
    Path(Path.home() / '.alphafold3/models_default_test').mkdir(parents=True, exist_ok=True)
    Path(Path.home() / '.alphafold3/databases_default_test').mkdir(parents=True, exist_ok=True)

    # Temporarily add them to defaults for this test script
    DEFAULT_MODEL_SEARCH_DIRS.insert(0, Path.home() / '.alphafold3/models_default_test')
    DEFAULT_DB_SEARCH_DIRS.insert(0, Path.home() / '.alphafold3/databases_default_test')


    try:
        resolved_model_dir = get_model_dir(cli_model_dir=mock_cli_model_dir)
        print(f"Resolved model directory: {resolved_model_dir}")
    except FileNotFoundError as e:
        print(f"Error resolving model dir: {e}")

    try:
        resolved_db_dirs = get_database_dirs(cli_db_dirs=mock_cli_db_dirs)
        print(f"Resolved database directories: {resolved_db_dirs}")

        if resolved_db_dirs:
            # Create a dummy file for find_database_file testing
            dummy_db_file = resolved_db_dirs[0] / "dummy_db_file.txt"
            with open(dummy_db_file, "w") as f:
                f.write("test")

            # Test find_database_file
            try:
                found_file = find_database_file(resolved_db_dirs, "dummy_db_file.txt")
                print(f"Found database file: {found_file}")
                os.remove(dummy_db_file) # Clean up

                # Test with ${DB_DIR}
                Path(resolved_db_dirs[0] / "bfd").mkdir(exist_ok=True)
                dummy_placeholder_file = resolved_db_dirs[0] / "bfd" / "bfd_test.fasta"
                with open(dummy_placeholder_file, "w") as f:
                    f.write("test_placeholder")

                found_placeholder_file = find_database_file(resolved_db_dirs, "${DB_DIR}/bfd/bfd_test.fasta")
                print(f"Found placeholder database file: {found_placeholder_file}")
                os.remove(dummy_placeholder_file) # Clean up
                os.rmdir(resolved_db_dirs[0] / "bfd")


            except FileNotFoundError as e:
                print(f"Error finding database file: {e}")

    except FileNotFoundError as e:
        print(f"Error resolving database dirs: {e}")

    # Clean up dummy directories
    if (Path.home() / '.alphafold3/models_default_test').exists():
        os.rmdir(Path.home() / '.alphafold3/models_default_test')
    if (Path.home() / '.alphafold3/databases_default_test').exists():
        os.rmdir(Path.home() / '.alphafold3/databases_default_test')

    print("\nFinished test.")
