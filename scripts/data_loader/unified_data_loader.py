import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import yaml


class DummyLanguageModel:
    """Minimal dummy language model for datasets that don't use text (e.g., TED Expressive)"""

    def __init__(self):
        self.SOS_token = 1
        self.EOS_token = 2
        self.n_words = 3

    def get_word_index(self, word):
        return 0


def _collate_aux_info(aux_infos):
    """Collate a sequence of aux_info dicts into a batched dict where possible.

    For mixed or non-collatable entries, values are kept as lists.
    """
    if not aux_infos or not aux_infos[0]:
        return {}

    aux_info_batch = {}
    keys = aux_infos[0].keys()

    for key in keys:
        try:
            aux_info_batch[key] = default_collate([aux[key] for aux in aux_infos])
        except Exception:
            aux_info_batch[key] = [aux[key] for aux in aux_infos]

    return aux_info_batch


def _make_per_dataset_collate_fn():
    """Create a collate function that turns a batch of 7-tuples into a standardized dict.

    Expected sample structure per dataset:
      (word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info)
    """

    def collate_fn(batch):
        if len(batch) == 0:
            # Maintain structure even for empty batches
            return {
                "word_seqs": torch.empty(0, dtype=torch.long),
                "extended_word_seqs": torch.empty(0, dtype=torch.long),
                "pose_seqs": torch.empty(0),
                "vec_seqs": torch.empty(0),
                "audios": torch.empty(0),
                "spectrograms": torch.empty(0),
                "aux_infos": {},
                "batch_size": 0,
            }

        word_seqs, extended_word_seqs, pose_seqs, vec_seqs, audios, spectrograms, aux_infos = zip(*batch)

        # Word sequences can be variable-length across samples. Try to collate, otherwise keep lists.
        try:
            word_seqs_collated = default_collate(word_seqs)
        except Exception:
            word_seqs_collated = list(word_seqs)
        try:
            extended_word_seqs_collated = default_collate(extended_word_seqs)
        except Exception:
            extended_word_seqs_collated = list(extended_word_seqs)

        batch_dict = {
            "word_seqs": word_seqs_collated,
            "extended_word_seqs": extended_word_seqs_collated,
            "pose_seqs": default_collate(pose_seqs),
            "vec_seqs": default_collate(vec_seqs),
            "audios": default_collate(audios),
            "spectrograms": default_collate(spectrograms),
            "aux_infos": _collate_aux_info(aux_infos),
            "batch_size": len(batch),
        }
        return batch_dict

    return collate_fn


class MultiDataLoaderWrapper:
    """Manages separate DataLoaders per dataset and provides balanced sampling per step.

    Each iteration returns a dict: { dataset_name: batch_dict }, where each batch_dict contains
    standardized keys: 'word_seqs', 'extended_word_seqs', 'pose_seqs', 'vec_seqs', 'audios',
    'spectrograms', 'aux_infos', and 'batch_size'.
    """

    def __init__(
        self,
        dataset_configs,
        samples_per_dataset_per_step=16,
        target_fps=15,
        n_poses=34,
        dataloader_overrides=None,
    ):
        """Create one DataLoader per dataset based on provided configs.

        Args:
            dataset_configs: List of dicts with at least keys: 'name', 'path', 'type', 'pose_dim'.
                             Optionally include per-dataset DataLoader kwargs (e.g., 'batch_size',
                             'num_workers', 'prefetch_factor', 'shuffle', 'pin_memory',
                             'persistent_workers', 'generator_seed').
            samples_per_dataset_per_step: Default batch size per dataset per training step.
            target_fps: Pose resampling FPS passed to datasets that need it (e.g., TED Expressive).
            n_poses: Number of poses per window (sequence length) for datasets that need it.
            dataloader_overrides: Optional dict mapping dataset name -> dict of DataLoader kwargs
                                  that override config defaults.
        """
        self.target_fps = target_fps
        self.n_poses = n_poses
        self.samples_per_dataset = samples_per_dataset_per_step
        self.dataloader_overrides = dataloader_overrides or {}

        self.datasets = {}
        self.dataloaders = {}
        self.iterators = {}
        self.dataset_sizes = {}
        self.dataset_batch_sizes = {}
        self.dataset_types = {}
        self.dataset_pose_dims = {}

        self._collate_fn = _make_per_dataset_collate_fn()

        successful = []
        for config in dataset_configs:
            name = config["name"]
            path = config["path"]
            dtype = config["type"]
            pose_dim = config.get("pose_dim")

            if not (path and os.path.exists(path)):
                print(f"Skipping dataset '{name}': path not found -> {path}")
                continue

            try:
                dataset = self._create_dataset(config)
                if len(dataset) == 0:
                    print(f"Skipping dataset '{name}': dataset is empty")
                    continue

                self.datasets[name] = dataset
                self.dataset_sizes[name] = len(dataset)
                self.dataset_types[name] = dtype
                self.dataset_pose_dims[name] = pose_dim

                # Build DataLoader with per-dataset overrides
                loader_kwargs = self._resolve_dataloader_kwargs(name, config)
                loader = DataLoader(dataset, collate_fn=self._collate_fn, **loader_kwargs)
                self.dataloaders[name] = loader
                self.dataset_batch_sizes[name] = loader_kwargs.get("batch_size", self.samples_per_dataset)

                # Infinite iterator
                self.iterators[name] = self._make_infinite_iterator(loader)

                successful.append(name)
                print(f"Loaded dataset '{name}' with {len(dataset)} samples (pose_dim={pose_dim})")
            except Exception as e:
                print(f"Failed to load dataset '{name}': {e}")
                import traceback

                print(traceback.format_exc())

        if not successful:
            raise ValueError("No datasets were successfully loaded for MultiDataLoaderWrapper")

        self.dataset_names = successful
        self._combined_iter = self._make_combined_iterator()

    def _resolve_dataloader_kwargs(self, dataset_name, config):
        """Resolve DataLoader kwargs from defaults, per-config, and overrides.

        Guards options that require workers > 0.
        """
        # Defaults
        kwargs = {
            "batch_size": config.get("batch_size", self.samples_per_dataset),
            "shuffle": config.get("shuffle", True),
            "num_workers": config.get("num_workers", 2),
            "pin_memory": config.get("pin_memory", True),
            "persistent_workers": config.get("persistent_workers", True),
        }

        if kwargs["num_workers"] <= 0:
            # Guard options that require workers
            kwargs["persistent_workers"] = False
            prefetch_factor = None
        else:
            prefetch_factor = config.get("prefetch_factor", 2)

        # Apply override dict (dataset-specific)
        if dataset_name in self.dataloader_overrides:
            for key, value in self.dataloader_overrides[dataset_name].items():
                kwargs[key] = value

        # If overrides changed num_workers, re-guard
        if kwargs["num_workers"] <= 0:
            kwargs["persistent_workers"] = False
            prefetch_factor = None

        # Only include prefetch_factor when valid
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor

        # Deterministic shuffling if a seed is supplied (per dataset)
        generator_seed = config.get("generator_seed")
        if (
            generator_seed is None
            and dataset_name in self.dataloader_overrides
            and "generator_seed" in self.dataloader_overrides[dataset_name]
        ):
            generator_seed = self.dataloader_overrides[dataset_name]["generator_seed"]

        if generator_seed is not None:
            g = torch.Generator()
            g.manual_seed(int(generator_seed))
            kwargs["generator"] = g

        return kwargs

    def _create_dataset(self, config):
        """Create a dataset instance based on config['type'].

        Supported types:
          - 'lmdb_new': Trinity / BEAT (LMDBDataset)
          - 'lmdb_expressive': TED Expressive (SpeechMotionDataset)
        """
        dtype = config["type"]

        if dtype == "lmdb_new":
            from scripts.data_loader.lmdb_data_loader_new import LMDBDataset

            return LMDBDataset(dataset_path=config["path"])

        if dtype == "lmdb_expressive":
            from scripts.data_loader.lmdb_data_loader_expressive import SpeechMotionDataset

            yaml_path = "./config/pose_diffusion_expressive.yml"
            with open(yaml_path, "r") as f:
                yaml_data = yaml.load(f, Loader=yaml.FullLoader)
            mean_pose = np.array(yaml_data["mean_pose"])  # ndarray for shape ops downstream
            mean_dir_vec = np.array(yaml_data["mean_dir_vec"])

            # Primary attempt
            try:
                dataset = SpeechMotionDataset(
                    lmdb_dir=config["path"],
                    n_poses=self.n_poses,
                    subdivision_stride=10,
                    pose_resampling_fps=self.target_fps,
                    mean_pose=mean_pose,
                    mean_dir_vec=mean_dir_vec,
                )
                dataset.set_lang_model(DummyLanguageModel())
                return dataset
            except Exception:
                # Fallback: disable speaker model if available
                try:
                    dataset = SpeechMotionDataset(
                        lmdb_dir=config["path"],
                        n_poses=self.n_poses,
                        subdivision_stride=10,
                        pose_resampling_fps=self.target_fps,
                        mean_pose=mean_pose,
                        mean_dir_vec=mean_dir_vec,
                        speaker_model=0,
                    )
                    dataset.set_lang_model(DummyLanguageModel())
                    return dataset
                except Exception as e2:
                    raise e2

        raise ValueError(f"Unknown dataset type: {dtype}")

    def _make_infinite_iterator(self, dataloader):
        while True:
            for batch in dataloader:
                yield batch

    def _make_combined_iterator(self):
        while True:
            combined = {}
            for name, it in self.iterators.items():
                combined[name] = next(it)
            yield combined

    def __iter__(self):
        return self._make_combined_iterator()

    def get_batch(self):
        """Return a single balanced batch dict across all datasets."""
        return next(self._combined_iter)

    def get_steps_per_epoch(self, mode="max", reference_name=None):
        """Derive steps per epoch based on dataset sizes and per-dataset batch sizes.

        Args:
            mode: One of {'max', 'min', 'name'}.
                  - 'max': number of steps to cover the largest dataset once
                  - 'min': number of steps to cover the smallest dataset once
                  - 'name': use a specific dataset by name (reference_name required)
            reference_name: Dataset name to use when mode == 'name'
        """
        steps = {}
        for name in self.dataset_names:
            size = self.dataset_sizes[name]
            bsz = self.dataset_batch_sizes[name]
            steps[name] = math.ceil(size / max(1, bsz))

        if mode == "max":
            return max(steps.values())
        if mode == "min":
            return min(steps.values())
        if mode == "name":
            if reference_name is None or reference_name not in steps:
                raise ValueError("reference_name must be a valid dataset name when mode='name'")
            return steps[reference_name]
        raise ValueError("mode must be one of {'max','min','name'}")

    def info(self):
        """Return a snapshot of datasets, sizes, and DataLoader settings."""
        return {
            "datasets": {
                name: {
                    "size": self.dataset_sizes[name],
                    "type": self.dataset_types[name],
                    "pose_dim": self.dataset_pose_dims[name],
                    "batch_size": self.dataset_batch_sizes[name],
                }
                for name in self.dataset_names
            },
            "names": list(self.dataset_names),
        }


def create_multi_dataloader(
    trinity_path=None,
    beat_path=None,
    ted_expressive_path=None,
    samples_per_dataset_per_step=16,
    target_fps=15,
    n_poses=34,
    dataloader_overrides=None,
):
    """Convenience factory for MultiDataLoaderWrapper.

    Only datasets with existing paths are included. Pose dimensions are set to
    defaults here but can be adjusted if needed.
    """
    dataset_configs = []

    if trinity_path and os.path.exists(trinity_path):
        dataset_configs.append(
            {
                "name": "trinity",
                "path": trinity_path,
                "type": "lmdb_new",
                "pose_dim": 129,
            }
        )

    if beat_path and os.path.exists(beat_path):
        dataset_configs.append(
            {
                "name": "beat",
                "path": beat_path,
                "type": "lmdb_new",
                "pose_dim": 177,
            }
        )

    if ted_expressive_path and os.path.exists(ted_expressive_path):
        dataset_configs.append(
            {
                "name": "ted_expressive",
                "path": ted_expressive_path,
                "type": "lmdb_expressive",
                "pose_dim": 114,
            }
        )

    if not dataset_configs:
        raise ValueError("At least one valid dataset path must be provided to create_multi_dataloader")

    return MultiDataLoaderWrapper(
        dataset_configs=dataset_configs,
        samples_per_dataset_per_step=samples_per_dataset_per_step,
        target_fps=target_fps,
        n_poses=n_poses,
        dataloader_overrides=dataloader_overrides,
    ) 