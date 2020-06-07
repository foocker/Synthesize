import numpy as np
import os
import copy
import logging
import collections
from typing import Any
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


class Checkpointer(object):
    """
    A checkpointer that can save/load model as well as extra checkpointable
    objects.
    """
    def __init__(self,
                 model: nn.Module,
                 save_dir: str = "",
                 *,
                 save_to_disk: bool = True,
                 **checkpointables: object):
        """
          Args:
              model (nn.Module): model.
              save_dir (str): a directory to save and find checkpoints.
              save_to_disk (bool): if True, save checkpoint to disk, otherwise
                  disable saving for this checkpointer.
              checkpointables (object): any checkpointable objects, i.e., objects
                  that have the `state_dict()` and `load_state_dict()` method. For
                  example, it can be used like
                  `Checkpointer(model, "dir", optimizer=optimizer)`.
          """
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.checkpointables = copy.copy(checkpointables)
        self.logger = logging.getLogger(__name__)
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk

    def save(self, name: str, **kwargs: dict):
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename    # ?
        with open(save_file, "wb") as f:    # may some problem?
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, path: str):
        if not path:
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            # may get the whole path
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)
        checkpoint = self._load_file(path)
        self._load_model(checkpoint)
        for key, obj in self.checkpointables.items():
            for key in checkpoint:
                self.logger.info("Loading {} from {}".format(key, path))
                obj.load_state_dict(checkpoint.pop(key))
        # return any further checkpoint dara
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            return ""
        return os.path.join(self.save_dir, last_saved)

    def get_all_checkpoint_files(self):
        all_model_checkpoints = [
            os.path.join(self.save_dir, file)
            for file in os.listdir(self.save_dir)
            if os.path.isfile(os.path.join(self.save_dir, file))
            and file.endswith(".pth")
        ]
        return all_model_checkpoints

    def resume_or_load(self, path: str, *, resume: bool = True):
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists.

        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
        return self.load(path)

    def tag_last_checkpoint(self, last_filename_basename: str):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename_basename)

    def _load_file(self, f: str):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint: Any):
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # work around https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.state_dict()
        for k in list(checkpoint_state_dict.keys()):
            for k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    self.logger.warning(
                        "'{}' has shape {} in the checkpoint but {} in the "
                        "model! Skipped.".format(
                            k, shape_checkpoint, shape_model
                        )
                    )
                    checkpoint_state_dict.pop(k)
        incompatible = self.model.load_state_dict(
            checkpoint_state_dict, strict=False
        )
        # can do some check for the incompatible

    def _convert_ndarray_to_tensor(self, state_dict: dict):
        """
                In-place convert all numpy arrays in the state_dict to torch tensor.
                Args:
                    state_dict (dict): a state-dict to be loaded to the model.
                """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError("Unsupported type found in checkpoint! {}: {}".format(k, type(v))
                                 )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)


class PeriodicCheckpointer:
    """
    Save checkpoints periodically. When `.step(iteration)` is called, it will
    execute `checkpointer.save` on the given checkpointer, if iteration is a
    multiple of period or if `max_iter` is reached.
    """
    def __init__(self, checkpointer: Any, period: int, max_iter: int = None):
        self.checkpointer = checkpointer
        self.period = int(period)
        self.max_iter = max_iter

    def step(self, iteration: int, **kwargs: Any):
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)
        if (iteration + 1) % self.period == 0:
            self.checkpointer.save("model_{:07d}".format(iteration), **additional_state)
        if iteration >= self.max_iter - 1:
            self.checkpointer.save("model_final", **additional_state)

    def save(self, name: str, **kwargs: Any):
        """
        Use this method to manually save checkpoints outside the schedule.
        """
        self.checkpointer.save(name, **kwargs)


def _strip_prefix_if_present(state_dict: collections.OrderedDict, prefix: str):
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(keys) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any...
    try:
        metadata = state_dict._metadata
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)






