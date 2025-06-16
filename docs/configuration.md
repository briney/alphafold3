# AlphaFold 3 Configuration

AlphaFold 3 uses a hierarchical system to configure paths for models and databases. This allows for flexibility in various environments.

## Configuration Hierarchy

Paths are resolved in the following order of precedence:

1.  **Command-Line Arguments:** Paths specified directly via CLI flags (e.g., `--model-dir`, `--db-dir`) will always take the highest priority.
2.  **Environment Variables:**
    *   `AF3_MODEL_DIR`: Path to the directory containing model parameters.
    *   `AF3_DB_DIRS`: A comma-separated list of paths to directories containing databases. (e.g., `/data/pdb,/data/uniref`)
3.  **Configuration File:** A JSON file that can be placed in one of the following locations:
    *   `~/.alphafold3/config.json`
    *   `~/.config/alphafold3/config.json` (on Linux/macOS)
    The first file found will be used. See the template below.
4.  **Default Locations:** If no configuration is provided through the above methods, AlphaFold 3 will search for models and databases in predefined default locations:
    *   **Models:**
        *   `~/.alphafold3/models/`
        *   `/opt/alphafold3/models/`
        *   `~/alphafold3/models/`
    *   **Databases:**
        *   `~/.alphafold3/databases/`
        *   `/opt/alphafold3/databases/`
        *   `~/alphafold3/databases/`

## Configuration File Template (`config.json`)

You can use the following template for your configuration file. Create it at one of the [specified locations](#configuration-file).

```json
{
  "model_dir": "/path/to/your/alphafold3/models",
  "database_dirs": [
    "/path/to/your/alphafold3/databases/main",
    "/path/to/your/other_alphafold3_databases"
  ]
}
```
Replace the example paths with the actual paths to your model and database directories.

## Database File Resolution

When looking for specific database files (e.g., `mgnify/mgy_clusters_2022_05.fa`), AlphaFold 3 will search within each configured database directory. Paths specified in the application (like `${DB_DIR}/bfd-first_non_consensus_sequences.fasta`) will have `${DB_DIR}` replaced by each of your configured database directories, and the first match found will be used. If a database path does not contain `${DB_DIR}`, it's treated as a relative path from each configured database directory.

## Path Validation

The system will check if the configured paths exist and are readable. If a required path (like the model directory) cannot be resolved through any of the mechanisms, an error will be raised with details on how to configure it. For database directories, if a path provided in the configuration is invalid, it will be ignored. If no valid database directories are found after checking all sources, an error will be raised.
