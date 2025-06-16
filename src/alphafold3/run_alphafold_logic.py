# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Logic for AlphaFold 3 structure prediction."""

from collections.abc import Callable, Sequence
import csv
import dataclasses
import datetime
import functools
import multiprocessing
import os
import pathlib
import shutil
import string
import textwrap
import time
import typing
from typing import overload

# Absolute imports for AlphaFold 3 modules
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import features
from alphafold3.model import model
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import utils
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np

_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
_DEFAULT_MODEL_DIR = _HOME_DIR / 'models'
_DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


def make_model_config(
    *,
    flash_attention_implementation: attention.Implementation = 'triton',
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
) -> model.Model.Config:
  """Returns a model config with some defaults overridden."""
  config = model.Model.Config()
  config.global_config.flash_attention_implementation = (
      flash_attention_implementation
  )
  config.heads.diffusion.eval.num_samples = num_diffusion_samples
  config.num_recycles = num_recycles
  config.return_embeddings = return_embeddings
  config.return_distogram = return_distogram
  return config


class ModelRunner:
  """Helper class to run structure prediction stages."""

  def __init__(
      self,
      config: model.Model.Config,
      device: jax.Device,
      model_dir: pathlib.Path,
  ):
    self._model_config = config
    self._device = device
    self._model_dir = model_dir

  @functools.cached_property
  def model_params(self) -> hk.Params:
    """Loads model parameters from the model directory."""
    return params.get_model_haiku_params(model_dir=self._model_dir)

  @functools.cached_property
  def _model(
      self,
  ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
    """Loads model parameters and returns a jitted model forward pass."""

    @hk.transform
    def forward_fn(batch):
      return model.Model(self._model_config)(batch)

    return functools.partial(
        jax.jit(forward_fn.apply, device=self._device), self.model_params
    )

  def run_inference(
      self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
  ) -> model.ModelResult:
    """Computes a forward pass of the model on a featurised example."""
    featurised_example = jax.device_put(
        jax.tree_util.tree_map(
            jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
        ),
        self._device,
    )

    result = self._model(rng_key, featurised_example)
    result = jax.tree.map(np.asarray, result)
    result = jax.tree.map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
        result,
    )
    result = dict(result)
    identifier = self.model_params['__meta__']['__identifier__'].tobytes()
    result['__identifier__'] = identifier
    return result

  def extract_inference_results(
      self,
      batch: features.BatchDict,
      result: model.ModelResult,
      target_name: str,
  ) -> list[model.InferenceResult]:
    """Extracts inference results from model outputs."""
    return list(
        model.Model.get_inference_result(
            batch=batch, result=result, target_name=target_name
        )
    )

  def extract_embeddings(
      self, result: model.ModelResult, num_tokens: int
  ) -> dict[str, np.ndarray] | None:
    """Extracts embeddings from model outputs."""
    embeddings = {}
    if 'single_embeddings' in result:
      embeddings['single_embeddings'] = result['single_embeddings'][
          :num_tokens
      ].astype(np.float16)
    if 'pair_embeddings' in result:
      embeddings['pair_embeddings'] = result['pair_embeddings'][
          :num_tokens, :num_tokens
      ].astype(np.float16)
    return embeddings or None

  def extract_distogram(
      self, result: model.ModelResult, num_tokens: int
  ) -> np.ndarray | None:
    """Extracts distogram from model outputs."""
    if 'distogram' not in result['distogram']:
      return None
    distogram = result['distogram']['distogram'][:num_tokens, :num_tokens, :]
    return distogram


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
  """Stores the inference results (diffusion samples) for a single seed.

  Attributes:
    seed: The seed used to generate the samples.
    inference_results: The inference results, one per sample.
    full_fold_input: The fold input that must also include the results of
      running the data pipeline - MSA and templates.
    embeddings: The final trunk single and pair embeddings, if requested.
    distogram: The token distance histogram, if requested.
  """

  seed: int
  inference_results: Sequence[model.InferenceResult]
  full_fold_input: folding_input.Input
  embeddings: dict[str, np.ndarray] | None = None
  distogram: np.ndarray | None = None


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
) -> Sequence[ResultsForSeed]:
  """Runs the full inference pipeline to predict structures for each seed."""

  print(f'Featurising data with {len(fold_input.rng_seeds)} seed(s)...')
  featurisation_start_time = time.time()
  ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
  featurised_examples = featurisation.featurise_input(
      fold_input=fold_input,
      buckets=buckets,
      ccd=ccd,
      verbose=True,
      ref_max_modified_date=ref_max_modified_date,
      conformer_max_iterations=conformer_max_iterations,
      resolve_msa_overlaps=resolve_msa_overlaps,
  )
  print(
      f'Featurising data with {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - featurisation_start_time:.2f} seconds.'
  )
  print(
      'Running model inference and extracting output structure samples with'
      f' {len(fold_input.rng_seeds)} seed(s)...'
  )
  all_inference_start_time = time.time()
  all_inference_results = []
  for seed, example in zip(fold_input.rng_seeds, featurised_examples):
    print(f'Running model inference with seed {seed}...')
    inference_start_time = time.time()
    rng_key = jax.random.PRNGKey(seed)
    result = model_runner.run_inference(example, rng_key)
    print(
        f'Running model inference with seed {seed} took'
        f' {time.time() - inference_start_time:.2f} seconds.'
    )
    print(f'Extracting inference results with seed {seed}...')
    extract_structures = time.time()
    inference_results = model_runner.extract_inference_results(
        batch=example, result=result, target_name=fold_input.name
    )
    num_tokens = len(inference_results[0].metadata['token_chain_ids'])
    embeddings = model_runner.extract_embeddings(
        result=result, num_tokens=num_tokens
    )
    distogram = model_runner.extract_distogram(
        result=result, num_tokens=num_tokens
    )
    print(
        f'Extracting {len(inference_results)} inference samples with'
        f' seed {seed} took {time.time() - extract_structures:.2f} seconds.'
    )

    all_inference_results.append(
        ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
            embeddings=embeddings,
            distogram=distogram,
        )
    )
  print(
      'Running model inference and extracting output structures with'
      f' {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - all_inference_start_time:.2f} seconds.'
  )
  return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Writes the input JSON to the output directory."""
  os.makedirs(output_dir, exist_ok=True)
  path = os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json')
  print(f'Writing model input JSON to {path}')
  with open(path, 'wt') as f:
    f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
  """Writes outputs to the specified output directory."""
  ranking_scores = []
  max_ranking_score = None
  max_ranking_result = None

  output_terms_path = pathlib.Path(alphafold3.cpp.__file__).parent.parent.parent / 'OUTPUT_TERMS_OF_USE.md' # Adjusted path
  output_terms = ""
  if output_terms_path.exists():
    output_terms = output_terms_path.read_text()
  else:
    print(f"Warning: OUTPUT_TERMS_OF_USE.md not found at {output_terms_path}")


  os.makedirs(output_dir, exist_ok=True)
  for results_for_seed in all_inference_results:
    seed = results_for_seed.seed
    for sample_idx, result in enumerate(results_for_seed.inference_results):
      sample_dir = os.path.join(output_dir, f'seed-{seed}_sample-{sample_idx}')
      os.makedirs(sample_dir, exist_ok=True)
      post_processing.write_output(
          inference_result=result,
          output_dir=sample_dir,
          name=f'{job_name}_seed-{seed}_sample-{sample_idx}',
      )
      ranking_score = float(result.metadata['ranking_score'])
      ranking_scores.append((seed, sample_idx, ranking_score))
      if max_ranking_score is None or ranking_score > max_ranking_score:
        max_ranking_score = ranking_score
        max_ranking_result = result

    if embeddings := results_for_seed.embeddings:
      embeddings_dir = os.path.join(output_dir, f'seed-{seed}_embeddings')
      os.makedirs(embeddings_dir, exist_ok=True)
      post_processing.write_embeddings(
          embeddings=embeddings,
          output_dir=embeddings_dir,
          name=f'{job_name}_seed-{seed}',
      )

    if (distogram := results_for_seed.distogram) is not None:
      distogram_dir = os.path.join(output_dir, f'seed-{seed}_distogram')
      os.makedirs(distogram_dir, exist_ok=True)
      distogram_path = os.path.join(
          distogram_dir, f'{job_name}_seed-{seed}_distogram.npz'
      )
      with open(distogram_path, 'wb') as f:
        np.savez_compressed(f, distogram=distogram.astype(np.float16))

  if max_ranking_result is not None:  # True iff ranking_scores non-empty.
    post_processing.write_output(
        inference_result=max_ranking_result,
        output_dir=output_dir,
        # The output terms of use are the same for all seeds/samples.
        terms_of_use=output_terms,
        name=job_name,
    )
    # Save csv of ranking scores with seeds and sample indices, to allow easier
    # comparison of ranking scores across different runs.
    with open(
        os.path.join(output_dir, f'{job_name}_ranking_scores.csv'), 'wt'
    ) as f:
      writer = csv.writer(f)
      writer.writerow(['seed', 'sample', 'ranking_score'])
      writer.writerows(ranking_scores)


def replace_db_dir(path_with_db_dir: str, db_dirs: Sequence[str]) -> str:
  """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
  template = string.Template(path_with_db_dir)
  if 'DB_DIR' in template.get_identifiers():
    for db_dir in db_dirs:
      path = template.substitute(DB_DIR=db_dir)
      if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f'{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}.'
    )
  if not os.path.exists(path_with_db_dir):
    raise FileNotFoundError(f'{path_with_db_dir} does not exist.')
  return path_with_db_dir


@overload
def process_fold_input(
    fold_input_obj: folding_input.Input, # Renamed from fold_input to avoid clash
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    force_output_dir: bool = False,
) -> folding_input.Input:
  ...


@overload
def process_fold_input(
    fold_input_obj: folding_input.Input, # Renamed from fold_input to avoid clash
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    force_output_dir: bool = False,
) -> Sequence[ResultsForSeed]:
  ...


def process_fold_input(
    fold_input_obj: folding_input.Input, # Renamed from fold_input to avoid clash
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    force_output_dir: bool = False,
) -> folding_input.Input | Sequence[ResultsForSeed]:
  """Runs data pipeline and/or inference on a single fold input."""
  print(f'\nRunning fold job {fold_input_obj.name}...')

  if not fold_input_obj.chains:
    raise ValueError('Fold input has no chains.')

  if (
      not force_output_dir
      and os.path.exists(output_dir)
      and os.listdir(output_dir)
  ):
    new_output_dir = (
        f'{output_dir}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    print(
        f'Output will be written in {new_output_dir} since {output_dir} is'
        ' non-empty.'
    )
    output_dir = new_output_dir
  else:
    print(f'Output will be written in {output_dir}')

  if data_pipeline_config is None:
    print('Skipping data pipeline...')
  else:
    print('Running data pipeline...')
    fold_input_obj = pipeline.DataPipeline(data_pipeline_config).process(fold_input_obj)

  write_fold_input_json(fold_input_obj, output_dir)
  if model_runner is None:
    print('Skipping model inference...')
    output = fold_input_obj
  else:
    print(
        f'Predicting 3D structure for {fold_input_obj.name} with'
        f' {len(fold_input_obj.rng_seeds)} seed(s)...'
    )
    all_inference_results = predict_structure(
        fold_input=fold_input_obj, # Passed as fold_input here
        model_runner=model_runner,
        buckets=buckets,
        ref_max_modified_date=ref_max_modified_date,
        conformer_max_iterations=conformer_max_iterations,
        resolve_msa_overlaps=resolve_msa_overlaps,
    )
    print(f'Writing outputs with {len(fold_input_obj.rng_seeds)} seed(s)...')
    write_outputs(
        all_inference_results=all_inference_results,
        output_dir=output_dir,
        job_name=fold_input_obj.sanitised_name(),
    )
    output = all_inference_results

  print(f'Fold job {fold_input_obj.name} done, output written to {output_dir}\n')
  return output


def run_alphafold_entrypoint(
    json_path: str | None = None,
    input_dir: str | None = None,
    output_dir_param: str = None, # Renamed to avoid clash with outer scope output_dir
    model_dir_param: str = _DEFAULT_MODEL_DIR.as_posix(), # Renamed
    run_data_pipeline: bool = True,
    run_inference: bool = True,
    jackhmmer_binary_path: str = shutil.which('jackhmmer'),
    nhmmer_binary_path: str = shutil.which('nhmmer'),
    hmmalign_binary_path: str = shutil.which('hmmalign'),
    hmmsearch_binary_path: str = shutil.which('hmmsearch'),
    hmmbuild_binary_path: str = shutil.which('hmmbuild'),
    db_dir: Sequence[str] = (_DEFAULT_DB_DIR.as_posix(),),
    small_bfd_database_path: str = '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    mgnify_database_path: str = '${DB_DIR}/mgy_clusters_2022_05.fa',
    uniprot_cluster_annot_database_path: str = '${DB_DIR}/uniprot_all_2021_04.fa',
    uniref90_database_path: str = '${DB_DIR}/uniref90_2022_05.fa',
    ntrna_database_path: str = '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    rfam_database_path: str = '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    rna_central_database_path: str = '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    pdb_database_path: str = '${DB_DIR}/mmcif_files',
    seqres_database_path: str = '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    jackhmmer_n_cpu: int = min(multiprocessing.cpu_count(), 8),
    nhmmer_n_cpu: int = min(multiprocessing.cpu_count(), 8),
    resolve_msa_overlaps: bool = True,
    max_template_date: str = '2021-09-30',
    conformer_max_iterations: int | None = None,
    jax_compilation_cache_dir: str | None = None,
    gpu_device: int = 0,
    buckets: Sequence[str] = ('256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072', '3584', '4096', '4608', '5120'),
    flash_attention_implementation: str = 'triton',
    num_recycles: int = 10,
    num_diffusion_samples: int = 5,
    num_seeds: int | None = None,
    save_embeddings: bool = False,
    save_distogram: bool = False,
    force_output_dir: bool = False,
):
  if jax_compilation_cache_dir is not None:
    jax.config.update(
        'jax_compilation_cache_dir', jax_compilation_cache_dir
    )

  if json_path is None == input_dir is None:
    raise ValueError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  if not run_inference and not run_data_pipeline:
    raise ValueError(
        'At least one of --run_inference or --run_data_pipeline must be'
        ' set to true.'
    )

  if input_dir is not None:
    fold_inputs = folding_input.load_fold_inputs_from_dir(
        pathlib.Path(input_dir)
    )
  elif json_path is not None:
    fold_inputs = folding_input.load_fold_inputs_from_path(
        pathlib.Path(json_path)
    )
  else:
    raise AssertionError( # Should be caught by the check above
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  # Make sure we can create the output directory before running anything.
  try:
    os.makedirs(output_dir_param, exist_ok=True)
  except OSError as e:
    print(f'Failed to create output directory {output_dir_param}: {e}')
    raise

  if run_inference:
    # Fail early on incompatible devices, but only if we're running inference.
    gpu_devices = jax.local_devices(backend='gpu')
    if gpu_devices:
      compute_capability = float(
          gpu_devices[gpu_device].compute_capability
      )
      if compute_capability < 6.0:
        raise ValueError(
            'AlphaFold 3 requires at least GPU compute capability 6.0 (see'
            ' https://developer.nvidia.com/cuda-gpus).'
        )
      elif 7.0 <= compute_capability < 8.0:
        xla_flags = os.environ.get('XLA_FLAGS')
        required_flag = '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
        if not xla_flags or required_flag not in xla_flags:
          raise ValueError(
              'For devices with GPU compute capability 7.x (see'
              ' https://developer.nvidia.com/cuda-gpus) the ENV XLA_FLAGS must'
              f' include "{required_flag}".'
          )
        if flash_attention_implementation != 'xla':
          raise ValueError(
              'For devices with GPU compute capability 7.x (see'
              ' https://developer.nvidia.com/cuda-gpus) the'
              ' --flash_attention_implementation must be set to "xla".'
          )

  notice = textwrap.wrap(
      'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
      ' parameters are only available under terms of use provided at'
      ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
      ' If you do not agree to these terms and are using AlphaFold 3 derived'
      ' model parameters, cancel execution of AlphaFold 3 inference with'
      ' CTRL-C, and do not use the model parameters.',
      break_long_words=False,
      break_on_hyphens=False,
      width=80,
  )
  print('\n' + '\n'.join(notice) + '\n')

  _max_template_date_obj = datetime.date.fromisoformat(max_template_date)
  if run_data_pipeline:
    expand_path = lambda x: replace_db_dir(x, db_dir)
    data_pipeline_config = pipeline.DataPipelineConfig(
        jackhmmer_binary_path=jackhmmer_binary_path,
        nhmmer_binary_path=nhmmer_binary_path,
        hmmalign_binary_path=hmmalign_binary_path,
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        small_bfd_database_path=expand_path(small_bfd_database_path),
        mgnify_database_path=expand_path(mgnify_database_path),
        uniprot_cluster_annot_database_path=expand_path(
            uniprot_cluster_annot_database_path
        ),
        uniref90_database_path=expand_path(uniref90_database_path),
        ntrna_database_path=expand_path(ntrna_database_path),
        rfam_database_path=expand_path(rfam_database_path),
        rna_central_database_path=expand_path(rna_central_database_path),
        pdb_database_path=expand_path(pdb_database_path),
        seqres_database_path=expand_path(seqres_database_path),
        jackhmmer_n_cpu=jackhmmer_n_cpu,
        nhmmer_n_cpu=nhmmer_n_cpu,
        max_template_date=_max_template_date_obj,
    )
  else:
    data_pipeline_config = None

  if run_inference:
    devices = jax.local_devices(backend='gpu')
    print(
        f'Found local devices: {devices}, using device {gpu_device}:'
        f' {devices[gpu_device]}'
    )

    print('Building model from scratch...')
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, flash_attention_implementation
            ),
            num_diffusion_samples=num_diffusion_samples,
            num_recycles=num_recycles,
            return_embeddings=save_embeddings,
            return_distogram=save_distogram,
        ),
        device=devices[gpu_device],
        model_dir=pathlib.Path(model_dir_param),
    )
    # Check we can load the model parameters before launching anything.
    print('Checking that model parameters can be loaded...')
    _ = model_runner.model_params
  else:
    model_runner = None

  num_fold_inputs_processed = 0 # Renamed
  for current_fold_input in fold_inputs: # Renamed
    if num_seeds is not None:
      print(f'Expanding fold job {current_fold_input.name} to {num_seeds} seeds')
      current_fold_input = current_fold_input.with_multiple_seeds(num_seeds)
    process_fold_input(
        fold_input_obj=current_fold_input, # Pass as fold_input_obj
        data_pipeline_config=data_pipeline_config,
        model_runner=model_runner,
        output_dir=os.path.join(output_dir_param, current_fold_input.sanitised_name()),
        buckets=tuple(int(b) for b in buckets), # Corrected: ensure buckets are integers
        ref_max_modified_date=_max_template_date_obj,
        conformer_max_iterations=conformer_max_iterations,
        resolve_msa_overlaps=resolve_msa_overlaps,
        force_output_dir=force_output_dir,
    )
    num_fold_inputs_processed += 1

  print(f'Done running {num_fold_inputs_processed} fold jobs.')
