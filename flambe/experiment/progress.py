import pickle
import os
import time
import json


class ProgressState:

    def __init__(self, name, save_path, dependency_dag, config, factories_num=0):
        self.fname = "state.pkl"

        self.config = config

        self.name = name
        self.save_path = save_path
        self.total = len(dependency_dag)

        self.done = 0

        # 0 if running local,
        # Number of factories in case running remotely
        self.factories_num = factories_num

        # 0 -> PENDING
        # 1 -> RUNNING
        # 2 -> SUCCESS
        # 3 -> FAILURE
        # 4 -> MASKED
        self.block_state = {}
        for b in dependency_dag.keys():
            self.block_state[b] = 0

        self.time_lapses = {}

        self.variants = {}

        self.finished = False

        self.prev_time = None

        self.dependency_dag = dependency_dag

        self._save()

    def checkpoint_start(self, block_id):
        self.prev_time = time.time()
        self.block_state[block_id] = 1

        self._save()

    def refresh(self):
        self.variants = {}
        for b in self.dependency_dag.keys():
            self.variants[b] = []
            block_path = os.path.join(self.save_path, b)
            if os.path.exists(block_path):
                variants = filter(
                    lambda x: os.path.isdir(os.path.join(block_path, x)), os.listdir(block_path)
                )
                for v in variants:
                    # Check if hparams were specified.
                    # hparams follows this syntax:
                    # "hparam1=value1,hparam2=value2"
                    if '=' in v:
                        v = v[:v.find("-") - 5]  # Substract year of date. TODO replace for regex
                        hparams = {}
                        for h in v.split(','):
                            k, v = h.split('=')
                            hparams[k] = v
                        self.variants[b].append(hparams)

    def checkpoint_end(self, block_id, block_success):
        curr_time = time.time()

        if self.prev_time:
            self.time_lapses[block_id] = curr_time - self.prev_time
            self.prev_time = None

        self.done += 1
        self.block_state[block_id] = 2 if block_success else 3

        self._save()

    def finish(self):
        self.finished = True
        self._save()

    def _save(self):
        with open(os.path.join(self.save_path, self.fname), 'wb') as f:
            pickle.dump(self, f)

    def toJSON(self):
        self.refresh()
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
