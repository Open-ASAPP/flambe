import pytest
import tempfile
import subprocess
import os


@pytest.mark.integration
def test_resources_config():
    config = """
!Experiment

name: random
save_path: {}

resources:
  train: {}

pipeline:
  dataset: !TabularDataset.from_path
    train_path: !@ train

"""

    with tempfile.TemporaryDirectory() as d, tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        test_data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        train_data = os.path.join(test_data_folder, 'dummy_tabular', 'train.csv')

        exp = config.format(d, train_data)
        f.write(exp)
        f.flush()
        ret = subprocess.run(['flambe', f.name])
        assert ret.returncode == 0
