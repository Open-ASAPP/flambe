import pytest
import tempfile
import subprocess
import os


@pytest.mark.integration
def test_text_classification_config():
    config = """
!Experiment

name: text_classification
save_path: {}

pipeline:
  b1: !Trainer
    dataset: !SSTDataset
      transform:
        text: !TextField
        label: !LabelField
    train_sampler: !BaseSampler
    val_sampler: !BaseSampler
    model: !TextClassifier
      embedder: !Embedder
        embedding: !torch.Embedding
          num_embeddings: !@ b1[dataset].text.vocab_size
          embedding_dim: 300
        encoder: !PooledRNNEncoder
          input_size: 300
          rnn_type: lstm
          n_layers: 2
          hidden_size: 256
      output_layer: !SoftmaxLayer
        input_size: !@ b1[model][embedder][encoder].rnn.hidden_size
        output_size: !@ b1[dataset].label.vocab_size
    loss_fn: !torch.NLLLoss
    metric_fn: !Accuracy
    optimizer: !torch.Adam
      params: !@ b1[model].trainable_params
    max_steps: 1
    iter_per_step: 1
"""

    with tempfile.TemporaryDirectory() as d, tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        exp = config.format(d)
        f.write(exp)
        f.flush()
        ret = subprocess.run(['flambe', f.name])
        assert ret.returncode == 0
