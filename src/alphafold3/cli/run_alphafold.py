# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction script CLI."""

import click
import multiprocessing
import pathlib
import shutil

from alphafold3.run_alphafold_logic import run_alphafold_entrypoint, _DEFAULT_MODEL_DIR, _DEFAULT_DB_DIR
from alphafold3.cli.main_cli import alphafold3 # Import the shared group

# Remove the local cli group definition, it's now imported.
# @click.group(name='alphafold3')
# def cli():
#     """AlphaFold 3 command line tools."""
#     pass

@alphafold3.command(name='predict') # Attach to the imported group
@click.option('--json-path', type=click.Path(exists=True, dir_okay=False), default=None, help='Path to the input JSON file.')
@click.option('--input-dir', type=click.Path(exists=True, file_okay=False), default=None, help='Path to the directory containing input JSON files.')
@click.option('--output-dir', type=click.Path(), required=True, help='Path to a directory where the results will be saved.')
@click.option('--model-dir', type=click.Path(exists=True, file_okay=False), default=_DEFAULT_MODEL_DIR.as_posix(), show_default=True, help='Path to the model to use for inference.')
@click.option('--run-data-pipeline/--no-run-data-pipeline', default=True, show_default=True, help='Whether to run the data pipeline on the fold inputs.')
@click.option('--run-inference/--no-run-inference', default=True, show_default=True, help='Whether to run inference on the fold inputs.')
@click.option('--jackhmmer-binary-path', type=click.Path(exists=True, dir_okay=False), default=shutil.which('jackhmmer'), show_default=True, help='Path to the Jackhmmer binary.')
@click.option('--nhmmer-binary-path', type=click.Path(exists=True, dir_okay=False), default=shutil.which('nhmmer'), show_default=True, help='Path to the Nhmmer binary.')
@click.option('--hmmalign-binary-path', type=click.Path(exists=True, dir_okay=False), default=shutil.which('hmmalign'), show_default=True, help='Path to the Hmmalign binary.')
@click.option('--hmmsearch-binary-path', type=click.Path(exists=True, dir_okay=False), default=shutil.which('hmmsearch'), show_default=True, help='Path to the Hmmsearch binary.')
@click.option('--hmmbuild-binary-path', type=click.Path(exists=True, dir_okay=False), default=shutil.which('hmmbuild'), show_default=True, help='Path to the Hmmbuild binary.')
@click.option('--db-dir', type=click.Path(exists=True, file_okay=False), multiple=True, default=[_DEFAULT_DB_DIR.as_posix()], show_default=True, help='Path to the directory containing the databases. Can be specified multiple times to search multiple directories in order.')
@click.option('--small-bfd-database-path', type=str, default='${DB_DIR}/bfd-first_non_consensus_sequences.fasta', show_default=True, help='Small BFD database path, used for protein MSA search.')
@click.option('--mgnify-database-path', type=str, default='${DB_DIR}/mgy_clusters_2022_05.fa', show_default=True, help='Mgnify database path, used for protein MSA search.')
@click.option('--uniprot-cluster-annot-database-path', type=str, default='${DB_DIR}/uniprot_all_2021_04.fa', show_default=True, help='UniProt database path, used for protein paired MSA search.')
@click.option('--uniref90-database-path', type=str, default='${DB_DIR}/uniref90_2022_05.fa', show_default=True, help='UniRef90 database path, used for MSA search.')
@click.option('--ntrna-database-path', type=str, default='${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta', show_default=True, help='NT-RNA database path, used for RNA MSA search.')
@click.option('--rfam-database-path', type=str, default='${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta', show_default=True, help='Rfam database path, used for RNA MSA search.')
@click.option('--rna-central-database-path', type=str, default='${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta', show_default=True, help='RNAcentral database path, used for RNA MSA search.')
@click.option('--pdb-database-path', type=str, default='${DB_DIR}/mmcif_files', show_default=True, help='PDB database directory with mmCIF files path, used for template search.')
@click.option('--seqres-database-path', type=str, default='${DB_DIR}/pdb_seqres_2022_09_28.fasta', show_default=True, help='PDB sequence database path, used for template search.')
@click.option('--jackhmmer-n-cpu', type=int, default=min(multiprocessing.cpu_count(), 8), show_default=True, help='Number of CPUs to use for Jackhmmer.')
@click.option('--nhmmer-n-cpu', type=int, default=min(multiprocessing.cpu_count(), 8), show_default=True, help='Number of CPUs to use for Nhmmer.')
@click.option('--resolve-msa-overlaps/--no-resolve-msa-overlaps', default=True, show_default=True, help='Whether to deduplicate unpaired MSA against paired MSA.')
@click.option('--max-template-date', type=str, default='2021-09-30', show_default=True, help='Maximum template release date to consider. Format: YYYY-MM-DD.')
@click.option('--conformer-max-iterations', type=int, default=None, help='Optional override for maximum number of iterations for RDKit conformer search.')
@click.option('--jax-compilation-cache-dir', type=click.Path(), default=None, help='Path to a directory for the JAX compilation cache.')
@click.option('--gpu-device', type=int, default=0, show_default=True, help='Optional override for the GPU device to use for inference.')
@click.option('--buckets', multiple=True, default=['256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072', '3584', '4096', '4608', '5120'], show_default=True, help='Token sizes for caching compilations.')
@click.option('--flash-attention-implementation', type=click.Choice(['triton', 'cudnn', 'xla']), default='triton', show_default=True, help="Flash attention implementation. 'triton' and 'cudnn' require Ampere GPUs or later. 'xla' is portable.")
@click.option('--num-recycles', type=int, default=10, show_default=True, help='Number of recycles to use during inference.')
@click.option('--num-diffusion-samples', type=int, default=5, show_default=True, help='Number of diffusion samples to generate.')
@click.option('--num-seeds', type=int, default=None, help='Number of seeds for inference. If set, a single seed must be in the input JSON.')
@click.option('--save-embeddings/--no-save-embeddings', default=False, show_default=True, help='Whether to save final trunk single and pair embeddings.')
@click.option('--save-distogram/--no-save-distogram', default=False, show_default=True, help='Whether to save the final distogram.')
@click.option('--force-output-dir/--no-force-output-dir', default=False, show_default=True, help='Force use of output directory even if it exists and is non-empty.')
def predict_command(
    json_path,
    input_dir,
    output_dir,
    model_dir,
    run_data_pipeline,
    run_inference,
    jackhmmer_binary_path,
    nhmmer_binary_path,
    hmmalign_binary_path,
    hmmsearch_binary_path,
    hmmbuild_binary_path,
    db_dir,
    small_bfd_database_path,
    mgnify_database_path,
    uniprot_cluster_annot_database_path,
    uniref90_database_path,
    ntrna_database_path,
    rfam_database_path,
    rna_central_database_path,
    pdb_database_path,
    seqres_database_path,
    jackhmmer_n_cpu,
    nhmmer_n_cpu,
    resolve_msa_overlaps,
    max_template_date,
    conformer_max_iterations,
    jax_compilation_cache_dir,
    gpu_device,
    buckets,
    flash_attention_implementation,
    num_recycles,
    num_diffusion_samples,
    num_seeds,
    save_embeddings,
    save_distogram,
    force_output_dir
):
    """Runs AlphaFold 3 structure prediction."""
    run_alphafold_entrypoint(
        json_path=json_path,
        input_dir=input_dir,
        output_dir_param=output_dir, # Passed as output_dir_param
        model_dir_param=model_dir,   # Passed as model_dir_param
        run_data_pipeline=run_data_pipeline,
        run_inference=run_inference,
        jackhmmer_binary_path=jackhmmer_binary_path,
        nhmmer_binary_path=nhmmer_binary_path,
        hmmalign_binary_path=hmmalign_binary_path,
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        db_dir=list(db_dir), # Convert tuple to list for consistency if needed by logic
        small_bfd_database_path=small_bfd_database_path,
        mgnify_database_path=mgnify_database_path,
        uniprot_cluster_annot_database_path=uniprot_cluster_annot_database_path,
        uniref90_database_path=uniref90_database_path,
        ntrna_database_path=ntrna_database_path,
        rfam_database_path=rfam_database_path,
        rna_central_database_path=rna_central_database_path,
        pdb_database_path=pdb_database_path,
        seqres_database_path=seqres_database_path,
        jackhmmer_n_cpu=jackhmmer_n_cpu,
        nhmmer_n_cpu=nhmmer_n_cpu,
        resolve_msa_overlaps=resolve_msa_overlaps,
        max_template_date=max_template_date,
        conformer_max_iterations=conformer_max_iterations,
        jax_compilation_cache_dir=jax_compilation_cache_dir,
        gpu_device=gpu_device,
        buckets=list(buckets), # Convert tuple to list
        flash_attention_implementation=flash_attention_implementation,
        num_recycles=num_recycles,
        num_diffusion_samples=num_diffusion_samples,
        num_seeds=num_seeds,
        save_embeddings=save_embeddings,
        save_distogram=save_distogram,
        force_output_dir=force_output_dir
    )

if __name__ == '__main__':
    alphafold3() # Call the imported group

main = predict_command
