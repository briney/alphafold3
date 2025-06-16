# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

import click
import shutil
import importlib
import importlib.metadata
from packaging import version as packaging_version
import jax
import os
import sys
import pathlib
import datetime
import json

# AlphaFold 3 specific imports
import run_alphafold # Main script for utilities
from alphafold3.common import folding_input
from alphafold3.data import pipeline as af3_pipeline # aliased to avoid conflict
from alphafold3.data import featurisation
from alphafold3.constants import chemical_components


def check_system_dependencies(verbose: bool, fix: bool): # Add verbose and fix arguments
    """Checks for essential system dependencies."""
    if verbose:
        click.echo("Checking system dependencies...")

    hmmer_binaries = [
        "jackhmmer",
        "nhmmer",
        "hmmalign",
        "hmmsearch",
        "hmmbuild",
    ]
    # One could add other system dependencies here in the future
    # e.g. {"tool_name": "kalign", "install_suggestion": "conda install -c bioconda kalign"}

    system_dependencies = {
        "HMMER": {
            "binaries": hmmer_binaries,
            "suggestion": (
                "Please install HMMER (version 3.1b2 or later) and ensure its tools are accessible. "
                "You can download HMMER from http://hmmer.org/download.html\n"
                "Alternatively, if you are using Conda, you can install HMMER by running:\n"
                "  conda install -c bioconda hmmer\n"
                "Ensure that the directory containing the HMMER binaries is added to your PATH environment variable."
            )
        }
        # Add other tools like kalign etc. here if needed in future
    }

    all_ok = True
    missing_overall = []

    for dep_name, dep_info in system_dependencies.items():
        missing_binaries = []
        for binary in dep_info["binaries"]:
            if shutil.which(binary) is None:
                missing_binaries.append(binary)
                missing_overall.append(binary)

        if missing_binaries:
            all_ok = False
            click.secho(f"  [FAIL] Missing {dep_name} binaries: {', '.join(missing_binaries)}", fg="red")
            conda_command = "conda install -c bioconda hmmer" # Specific to HMMER for now
            if dep_name == "HMMER":
                click.echo(f"    Suggestion: Install HMMER (e.g., using Conda: `{conda_command}` ) or from http://hmmer.org/download.html")
                click.echo(f"    Ensure the installation directory is in your PATH.")
                if fix:
                    click.echo(click.style(f"    --fix mode: You could try running: `{conda_command}`", bold=True))
            else: # Generic suggestion for other potential system deps
                 click.echo(f"    Suggestion: {dep_info['suggestion']}")
                 if fix:
                    click.echo(click.style(f"    --fix mode: Auto-remediation for {dep_name} is not implemented. Please follow suggestions above.", bold=True))
        elif verbose: # Only show [OK] messages if verbose
            click.secho(f"  [OK] {dep_name} binaries found: {', '.join(dep_info['binaries'])}", fg="green")

    if not missing_overall and verbose: # Overall success summary only if verbose
        click.secho("All system dependencies satisfied.", fg="green")
    elif not all_ok: # Failure summary should always print
        click.secho("Some system dependencies are missing. Please see suggestions above.", fg="yellow")

    return all_ok


def check_python_packages(verbose: bool, fix: bool): # Add verbose and fix arguments
    """Checks for required Python packages and their versions."""
    if verbose:
        click.echo("Checking Python package dependencies...")

    REQUIRED_PYTHON_PACKAGES = {
        "absl-py": "2.1.0",
        "dm-haiku": "0.0.13",
        "dm-tree": "0.1.8",
        "jax": "0.4.34",
        "jax-triton": "0.2.0",
        "jaxtyping": "0.2.34",
        "numpy": "2.1.3",
        "rdkit": "2024.3.5",
        "triton": "3.1.0",
        "tqdm": "4.67.0",
        "typeguard": "2.13.3",
        "zstandard": "0.23.0",
        "click": None,
        "requests": None,
        "packaging": None, # Required for version parsing
    }

    all_ok = True
    for package_name, required_version_str in REQUIRED_PYTHON_PACKAGES.items():
        try:
            installed_version_str = importlib.metadata.version(package_name)
            if required_version_str:
                installed_version = packaging_version.parse(installed_version_str)
                required_version = packaging_version.parse(required_version_str)
                if installed_version < required_version:
                    all_ok = False
                    click.secho(f"  [FAIL] {package_name} is outdated (found {installed_version_str}, require {required_version_str})", fg="red")
            upgrade_command = f"pip install --upgrade \"{package_name}=={required_version_str}\""
            click.echo(f"    Suggestion: {upgrade_command}")
                    if fix:
                click.echo(click.style(f"    --fix mode: You could try running: `{upgrade_command}`", bold=True))
                elif verbose: # Only show detailed [OK] messages if verbose
                    click.secho(f"  [OK] {package_name} version {installed_version_str} (meets requirement {required_version_str})", fg="green")
            elif verbose: # Only show detailed [OK] messages if verbose (package found, no specific version)
                click.secho(f"  [OK] {package_name} version {installed_version_str} (found)", fg="green")

        except importlib.metadata.PackageNotFoundError:
            all_ok = False
            suggestion_command = f"pip install \"{package_name}"
            if required_version_str:
                suggestion_command += f"=={required_version_str}\""
            else:
                suggestion_command += "\""
            click.secho(f"  [FAIL] {package_name} is not installed.", fg="red") # FAIL messages always print
            click.echo(f"    Suggestion: {suggestion_command}") # Suggestions for FAIL always print
            if fix:
                click.echo(click.style(f"    --fix mode: You could try running: `{suggestion_command}`", bold=True))
        except Exception as e: # Catch other potential errors during version check
            all_ok = False
            click.secho(f"  [ERROR] Could not check version for {package_name}: {e}", fg="red") # ERROR messages always print


    if all_ok and verbose: # Overall success summary only if verbose
        click.secho("All Python package dependencies are satisfied.", fg="green")
    elif not all_ok: # Failure summary should always print
        click.secho("Some Python package dependencies are missing or outdated. Please see suggestions above.", fg="yellow")

    return all_ok


def check_gpu_cuda(verbose: bool):
    """Checks for GPU availability and CUDA setup."""
    if verbose:
        click.echo("Checking GPU/CUDA availability...")

    try:
        devices = jax.local_devices()
        gpus_found = []
        if not devices and verbose: # Only print "No JAX devices found." if verbose, as it might be normal for CPU-only setups.
            click.echo("  No JAX devices found.")

        for device in devices:
            if device.platform.lower() == 'gpu':
                gpus_found.append(device)
            if verbose: # Detailed device listing only if verbose
                click.echo(f"  Found device: {device.platform} - {device.device_kind} (id={device.id})")

        if gpus_found:
            click.secho(f"  [OK] Found {len(gpus_found)} GPU(s) usable by JAX:", fg="green") # Always print if GPUs are found
            if verbose: # Details of GPUs only if verbose
                for i, gpu in enumerate(gpus_found):
                    click.echo(f"    GPU {i}: {gpu.device_kind}")
            try:
                backend = jax.lib.xla_bridge.get_backend()
                cuda_version = backend.platform_version
                #platform_version might be a string like "conn_version=12.2, cudart_version=12.2, driver_version=535.129.03"
                #or just the version number depending on JAX internal changes.
                if verbose: # CUDA version detail only if verbose
                    click.echo(f"    JAX platform version (includes CUDA info if applicable): {cuda_version}")
            except Exception as e:
                if verbose: # CUDA version error detail only if verbose
                    click.echo(f"    Could not determine JAX CUDA version details: {e}")
            return True
        else: # No GPUs found
            click.secho("  [FAIL] No GPUs found or JAX is not configured to use GPUs.", fg="red") # Always print FAIL
            click.echo("    Suggestions:") # Always print suggestions
            click.echo("      - Ensure NVIDIA drivers are correctly installed and up to date.")
            click.echo("      - Ensure the CUDA toolkit is installed and compatible with your NVIDIA drivers and JAX version.")
            click.echo("      - Verify that JAX was installed with CUDA support. You might need to reinstall it, e.g.:")
            click.echo("        `pip install --upgrade jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`")
            click.echo("        (Adjust 'cuda12' based on your CUDA version if necessary).")
            click.echo("      - Check JAX documentation for GPU troubleshooting: https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu")
            return False

    except Exception as e:
        click.secho(f"  [ERROR] An error occurred while checking for GPUs: {e}", fg="red")
        click.echo("    This might indicate an issue with your JAX installation or system configuration.")
        return False

def check_model_database_availability(verbose: bool):
    """Checks for model and database file availability."""
    if verbose:
        click.echo("Checking model and (miniature) database availability...")

    all_found = True
    model_dir_ok = True

    # 1. Check Model Directory (alphafold3_weights)
    MODEL_DIR = pathlib.Path("alphafold3_weights")
    # For a more thorough check, one might list specific essential files here
    # EXPECTED_MODEL_FILES = [MODEL_DIR / "params_model_1.npz", MODEL_DIR / "params_model_2.npz"]

    if MODEL_DIR.exists() and MODEL_DIR.is_dir():
        if list(MODEL_DIR.glob('*')):
            if verbose: # OK message for model dir only if verbose
                click.secho(f"  [OK] Model directory found and appears to contain files: {MODEL_DIR}", fg="green")
        else:
            model_dir_ok = False
            all_found = False
            click.secho(f"  [FAIL] Model directory found but is empty: {MODEL_DIR}", fg="red") # FAIL always prints
            click.echo(f"    Suggestion: Ensure models have been downloaded to '{MODEL_DIR}'.") # Suggestion always prints
            click.echo(f"    Run 'alphafold3-fetch-databases' or check documentation for model download instructions.")
    else:
        model_dir_ok = False
        all_found = False
        click.secho(f"  [FAIL] Model directory not found: {MODEL_DIR}", fg="red") # FAIL always prints
        click.echo(f"    Suggestion: Ensure models have been downloaded to '{MODEL_DIR}'.") # Suggestion always prints
        click.echo(f"    Run 'alphafold3-fetch-databases' script or check documentation for model download instructions.")

    # 2. Check Miniature Database Files (for running tests/health_check functionality test)
    MINI_DB_DIR = pathlib.Path("src/alphafold3/test_data/miniature_databases")
    # Check a couple of representative files/dirs from the ls output
    EXPECTED_MINI_DB_ITEMS = [
        MINI_DB_DIR / "uniref90__subsampled_1000.fasta",
        MINI_DB_DIR / "pdb_mmcif", # Check directory
        MINI_DB_DIR / "pdb_mmcif/5y2e.cif" # Check a file within directory
    ]
    # This check is more for the health_check's own functionality test later
    # rather than a full user database check.

    mini_db_ok = True
    if not (MINI_DB_DIR.exists() and MINI_DB_DIR.is_dir()):
        mini_db_ok = False
        all_found = False
        click.secho(f"  [FAIL] Miniature database directory not found: {MINI_DB_DIR}", fg="red") # FAIL always prints
        click.echo(f"    Suggestion: This directory should be part of the AlphaFold 3 source code. Please verify your installation integrity.") # Suggestion always prints
    else:
        for item_path in EXPECTED_MINI_DB_ITEMS:
            if item_path.exists():
                if verbose: # OK for individual DB items only if verbose
                    click.secho(f"  [OK] Found miniature database item: {item_path}", fg="green")
            else:
                mini_db_ok = False
                all_found = False
                click.secho(f"  [FAIL] Missing miniature database item: {item_path}", fg="red") # FAIL always prints
                click.echo(f"    Suggestion: This item should be part of the AlphaFold 3 source code test_data. Please verify your installation integrity.") # Suggestion always prints

    if verbose: # Section summaries for verbose mode
        if model_dir_ok:
            click.secho("Model directory check passed (verbose).", fg="green")
        if mini_db_ok:
            click.secho("Miniature database check passed (verbose).", fg="green")

    if not all_found: # Overall failure summary for this section always prints
        click.secho("Some model or essential miniature database files are missing. Please see suggestions above.", fg="yellow")
    elif verbose and all_found: # Overall success summary only if verbose
        click.secho("Essential model and miniature database files appear to be available.", fg="green")

    return all_found

def run_quick_functionality_test(verbose: bool):
    """Runs a minimal test case for basic functionality verification."""
    if verbose:
        click.echo("Running quick functionality test (this may take a moment)...")

    # Check if model directory exists and has files first (as a prerequisite)
    MODEL_DIR = pathlib.Path("alphafold3_weights")
    if not (MODEL_DIR.exists() and MODEL_DIR.is_dir() and list(MODEL_DIR.glob('*'))):
        click.secho("  [SKIP] Quick functionality test: Model directory is missing or empty.", fg="yellow") # SKIP always prints
        click.echo(f"    Ensure models are downloaded to '{MODEL_DIR}'. This test cannot run without models.") # Suggestion always prints
        return False

    # Check if a JAX device (preferably GPU) is available
    target_device = None
    try:
        if not jax.local_devices():
            click.secho("  [FAIL] Quick functionality test: No JAX devices found.", fg="red") # FAIL always prints
            return False
        # Prioritize GPU if available
        if jax.local_devices(backend='gpu'):
            target_device = jax.local_devices(backend='gpu')[0]
        else:
            target_device = jax.local_devices()[0]
        if verbose: # JAX device info only if verbose
            click.echo(f"  Using JAX device: {target_device.platform} - {target_device.device_kind}")
    except Exception as e:
        click.secho(f"  [FAIL] Quick functionality test: Error initializing JAX device: {e}", fg="red") # FAIL always prints
        return False

    # Setup: Paths and Configs
    base_path = pathlib.Path(__file__).parent.parent / "test_data" / "miniature_databases"
    MINI_DB_ROOT = base_path.resolve()

    if not MINI_DB_ROOT.exists():
        click.secho(f"  [FAIL] Quick functionality test: Miniature database directory not found at {MINI_DB_ROOT}", fg="red") # FAIL always prints
        click.echo(f"    This directory is essential for the test and should be part of the AF3 source.") # Suggestion always prints
        return False

    # Ensure all HMMER tools are found
    hmmer_binaries = ["jackhmmer", "nhmmer", "hmmalign", "hmmsearch", "hmmbuild"]
    found_hmmer_paths = {}
    missing_hmmer_for_test = []
    for binary_name in hmmer_binaries:
        path = shutil.which(binary_name)
        if path:
            found_hmmer_paths[f"{binary_name}_binary_path"] = path
        else:
            missing_hmmer_for_test.append(binary_name)

    if missing_hmmer_for_test:
        click.secho(f"  [FAIL] Quick functionality test: Missing HMMER binaries required for DataPipeline: {', '.join(missing_hmmer_for_test)}", fg="red") # FAIL always prints
        click.echo(f"    Ensure HMMER tools are installed and in PATH. These are checked separately but also vital for this test.") # Suggestion always prints
        return False

    data_pipeline_config = af3_pipeline.DataPipelineConfig(
        jackhmmer_binary_path=found_hmmer_paths.get("jackhmmer_binary_path"),
        nhmmer_binary_path=found_hmmer_paths.get("nhmmer_binary_path"),
        hmmalign_binary_path=found_hmmer_paths.get("hmmalign_binary_path"),
        hmmsearch_binary_path=found_hmmer_paths.get("hmmsearch_binary_path"),
        hmmbuild_binary_path=found_hmmer_paths.get("hmmbuild_binary_path"),
        small_bfd_database_path=MINI_DB_ROOT / "bfd-first_non_consensus_sequences__subsampled_1000.fasta",
        mgnify_database_path=MINI_DB_ROOT / "mgy_clusters__subsampled_1000.fa",
        uniprot_cluster_annot_database_path=MINI_DB_ROOT / "uniprot_all__subsampled_1000.fasta",
        uniref90_database_path=MINI_DB_ROOT / "uniref90__subsampled_1000.fasta",
        ntrna_database_path=MINI_DB_ROOT / "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq__subsampled_1000.fasta",
        rfam_database_path=MINI_DB_ROOT / "rfam_14_4_clustered_rep_seq__subsampled_1000.fasta",
        rna_central_database_path=MINI_DB_ROOT / "rnacentral_active_seq_id_90_cov_80_linclust__subsampled_1000.fasta",
        pdb_database_path=MINI_DB_ROOT / "pdb_mmcif",
        seqres_database_path=MINI_DB_ROOT / "pdb_seqres_2022_09_28__subsampled_1000.fasta",
        max_template_date=datetime.date(2023, 1, 1),
    )

    test_input_dict = {
        'name': 'health_check_test_peptide',
        'modelSeeds': [1234],
        'sequences': [{'protein': {'id': 'A', 'sequence': 'GGC'}}],
        'dialect': folding_input.JSON_DIALECT,
        'version': folding_input.JSON_VERSION,
    }
    fold_input_obj = folding_input.Input.from_json(json.dumps(test_input_dict))

    try:
        if verbose: click.echo("  Initializing DataPipeline...") # Step messages only if verbose
        data_pipeline = af3_pipeline.DataPipeline(data_pipeline_config)

        if verbose: click.echo(f"  Processing input: {fold_input_obj.name}...") # Step messages only if verbose
        full_fold_input = data_pipeline.process(fold_input_obj)

        if verbose: click.echo("  Featurizing input...") # Step messages only if verbose
        ccd_data = chemical_components.cached_ccd()
        featurised_example = featurisation.featurise_input(
            full_fold_input, ccd=ccd_data, buckets=None
        )

        if verbose: click.echo("  Initializing ModelRunner...") # Step messages only if verbose
        model_config = run_alphafold.make_model_config(
            flash_attention_implementation='triton',
        )
        model_runner = run_alphafold.ModelRunner(
            config=model_config, device=target_device, model_dir=MODEL_DIR,
        )

        if verbose: click.echo("  Running inference (on minimal input)...") # Step messages only if verbose
        result = model_runner.run_inference(featurised_example[0], jax.random.PRNGKey(0))

        if result and 'predicted_structure' in result and result['predicted_structure'] is not None:
            _ = result['predicted_structure'].coords
            if verbose: click.echo(f"  Inference result keys: {list(result.keys())}") # Result details only if verbose
            click.secho("  [OK] Quick functionality test completed successfully.", fg="green") # OK always prints
            return True
        else:
            click.secho("  [FAIL] Quick functionality test ran but produced an unexpected or empty result.", fg="red") # FAIL always prints
            if verbose and result: click.echo(f"  Inference result keys found: {list(result.keys())}") # Result details only if verbose
            elif verbose: click.echo("  Inference result was None or did not contain 'predicted_structure'.")
            return False

    except Exception as e:
        click.secho(f"  [FAIL] Quick functionality test encountered an error: {e}", fg="red") # FAIL always prints
        if verbose: # Traceback only if verbose
            import traceback
            click.echo("    Traceback:")
            click.echo(traceback.format_exc())
        click.echo("    This may indicate issues with model loading, data processing, or core AlphaFold 3 execution.") # Suggestion always prints
        return False

@click.command()
@click.option('--verbose', is_flag=True, help="Enable verbose output.")
@click.option('--quiet', is_flag=True, help="Suppress all output except errors.")
@click.option('--fix', is_flag=True, help="Attempt to automatically fix common issues.")
def main_cli(verbose: bool, quiet: bool, fix: bool):
    """AlphaFold 3 Health Check Tool"""
    original_stdout = sys.stdout
    devnull = None # Define devnull here to ensure it's in scope for finally block if needed
    if quiet:
        # Suppress all echo calls if quiet mode is enabled
        # Ensure this is done correctly to not interfere with underlying processes if they write to stdout directly
        # For click.echo, this should be fine.
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull


    if not quiet: # This condition ensures that this message itself isn't suppressed
        click.echo("Starting AlphaFold 3 Health Check...")

    all_checks_passed = True

    system_deps_ok = check_system_dependencies(verbose=verbose, fix=fix)
    if not system_deps_ok:
        all_checks_passed = False

    python_pkgs_ok = check_python_packages(verbose=verbose, fix=fix)
    if not python_pkgs_ok:
        all_checks_passed = False

    gpu_cuda_ok = check_gpu_cuda(verbose=verbose)
    if not gpu_cuda_ok:
        all_checks_passed = False

    model_db_ok = check_model_database_availability(verbose=verbose)
    if not model_db_ok:
        all_checks_passed = False
        click.echo("    Skipping quick functionality test as models or essential databases are missing.")
    else: # Only run the test if models and mini dbs seem okay
        quick_test_ok = run_quick_functionality_test(verbose=verbose)
        if not quick_test_ok:
            all_checks_passed = False

    if not quiet:
        if all_checks_passed:
            click.secho("AlphaFold 3 Health Check finished. All checks passed.", fg="green") # Final message updated
        else:
            click.secho("AlphaFold 3 Health Check finished. Some checks failed.", fg="red")

        # Restore stdout
        if devnull: # Check if devnull was actually opened
            sys.stdout = original_stdout # Restore original stdout
            devnull.close() # Close the null device


if __name__ == '__main__':
    main_cli()
