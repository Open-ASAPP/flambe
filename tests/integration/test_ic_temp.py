import pytest
import tempfile
import subprocess


@pytest.mark.integration
def test_image_classification_config():
    config = """
!Experiment

name: image_classification
save_path: {}

pipeline:
  model: !ImageClassifier
    encoder: !CNNEncoder
      input_channels: 1
      channels: [1]
      kernel_size: [5]
    output_layer: !MLPEncoder
      input_size: 576
      n_layers: 2
      output_size: 10
      output_activation: !torch.LogSoftmax
      hidden_size: 128

  train: !Trainer
    dataset: !MNISTDataset
    train_sampler: !BaseSampler
      batch_size: 64
    val_sampler: !BaseSampler
    model: !@ model
    loss_fn: !torch.NLLLoss
    metric_fn: !Accuracy
    optimizer: !torch.Adam
      params: !@ train[model].trainable_params
    max_steps: 1
    iter_per_step: 1
"""

    with tempfile.TemporaryDirectory() as d, tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        exp = config.format(d)
        f.write(exp)
        f.flush()
        ret = subprocess.run(['flambe', f.name])
        assert ret.returncode == 0
