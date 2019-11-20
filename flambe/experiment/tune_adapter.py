# from __future__ import annotations
import os
import pickle
from six import string_types
from typing import Dict, Optional
from copy import deepcopy
from collections import OrderedDict
import shutil

import ray

from flambe.compile import load_state_from_file, Schema, Component
from flambe.compile.extensions import setup_default_modules, import_modules
from flambe.experiment import utils
from flambe.logging import TrialLogging


class TuneAdapter(ray.tune.Trainable):
    """Adapter to the tune.Trainable inteface."""

    def _setup(self, config: Dict):
        """Subclasses should override this for custom initialization."""
        # Set this flag to False, if we find an error, or reduce
        self.name = config['name']
        self.run_flag = True
        custom_modules = config['custom_modules']
        setup_default_modules()
        import_modules(custom_modules)
        # Get the current computation block
        target_block_id = config['to_run']
        self.block_id = target_block_id
        # Update the schemas with the configuration
        schemas: Dict[str, Schema] = Schema.deserialize(config['schemas'])
        schemas_copy = deepcopy(schemas)
        global_vars = config['global_vars']
        self.verbose = config['verbose']
        self.hyper_params = config['hyper_params']
        self.debug = config['debug']

        with TrialLogging(log_dir=self.logdir,
                          verbose=self.verbose,
                          console_prefix=self.block_id,
                          hyper_params=self.hyper_params,
                          capture_warnings=True):

            # Compile, activate links, and load checkpoints
            filled_schemas: Dict = OrderedDict()
            for block_id, schema_block in schemas.items():

                block_params = config['params'][block_id]

                utils.update_schema_with_params(schemas_copy[block_id], block_params)

                # First activate links from previous blocks in the
                # pipeline
                utils.update_link_refs(schemas_copy, block_id, global_vars)
                block: Component = schemas_copy[block_id]()
                filled_schemas[block_id] = schemas_copy[block_id]

                if block_id in config['checkpoints']:
                    # Get the block hash
                    needed_set = utils.extract_needed_blocks(schemas, block_id, global_vars)
                    needed_blocks = ((k, v) for k, v in filled_schemas.items() if k in needed_set)
                    block_hash = repr(OrderedDict(needed_blocks))

                    # Check the mask, if it's False then we end
                    # immediately
                    mask_value = config['checkpoints'][block_id]['mask'][block_hash]
                    if mask_value is False:
                        self.run_flag = False
                        return

                    # There should be a checkpoint
                    checkpoint = config['checkpoints'][block_id]['paths'][block_hash]
                    state = load_state_from_file(checkpoint)
                    block.load_state(state)

                # Holding compiled objects alongside schemas is okay
                # but not fully expressed in our type annotations.
                # TODO: fix this in our utils type annotations
                schemas_copy[block_id] = block  # type: ignore

        # If everything went well, just compile
        self.block = schemas_copy[target_block_id]

        # Add tb prefix to computables in case multiple plots are
        # requested
        if not config['merge_plot']:
            self.block.tb_log_prefix = self.name

    def save(self, checkpoint_dir: Optional[str] = None) -> str:
        """Override to replace checkpoint."""
        checkpoint_dir = os.path.join(checkpoint_dir or self.logdir, "checkpoint")
        if not self.run_flag:
            return checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint = self._save(checkpoint_dir)
        saved_as_dict = False
        if isinstance(checkpoint, string_types):
            if (not checkpoint.startswith(checkpoint_dir) or checkpoint == checkpoint_dir):
                raise ValueError(
                    "The returned checkpoint path must be within the "
                    "given checkpoint dir {}: {}".format(
                        checkpoint_dir, checkpoint))
            if not os.path.exists(checkpoint):
                raise ValueError(
                    "The returned checkpoint path does not exist: {}".format(
                        checkpoint))
            checkpoint_path = checkpoint
        elif isinstance(checkpoint, dict):
            saved_as_dict = True
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)
        else:
            raise ValueError(
                "`_save` must return a dict or string type: {}".format(
                    str(type(checkpoint))))
        with open(checkpoint_path + ".tune_metadata", "wb") as f:
            pickle.dump({
                "experiment_id": self._experiment_id,
                "iteration": self._iteration,
                "timesteps_total": self._timesteps_total,
                "time_total": self._time_total,
                "episodes_total": self._episodes_total,
                "saved_as_dict": saved_as_dict
            }, f)
        return checkpoint_path

    def _train(self) -> Dict:
        """Subclasses should override this to implement train()."""
        if self.run_flag is False:
            return {'done': True}

        with TrialLogging(log_dir=self.logdir,
                          verbose=self.verbose,
                          console_prefix=self.block_id,
                          hyper_params=self.hyper_params,
                          capture_warnings=True):
            # Tune uses "done" instead of "continue" flag, so reverse
            # the boolean
            done = not self.block.run()
            metric = self.block.metric()

        report = {'done': done}
        if metric is not None:
            report['episode_reward_mean'] = metric

        return report

    def _save(self, checkpoint_dir: str) -> str:
        """Subclasses should override this to implement save()."""
        path = os.path.join(checkpoint_dir, "checkpoint.flambe")
        self.block.save(path, overwrite=True)
        return path

    def _restore(self, checkpoint: str) -> None:
        """Subclasses should override this to implement restore()."""
        state = load_state_from_file(checkpoint)
        self.block.load_state(state)

    def _stop(self):
        """Subclasses should override this for any cleanup on stop."""
        if not self.run_flag:
            shutil.rmtree(self.logdir)

        if hasattr(self, 'block') and self.block is not None:
            del self.block
