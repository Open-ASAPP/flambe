import pytest
import flambe
from flambe.nlp.classification import TextClassifier
from flambe.compile import extensions
from tempfile import TemporaryDirectory as tmpdir
from tempfile import NamedTemporaryFile as tmpfile
import subprocess
import os


def module_equals(model1, model2):
    """Check if 2 pytorch modules have the same weights

    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def test_exporter_builder():
    with tmpdir() as d, tmpdir() as d2, tmpfile(mode="w", suffix=".yaml") as f, tmpfile(mode="w", suffix=".yaml") as f2:
        # First run an experiment
        exp = """
!Experiment

name: exporter
save_path: {}

pipeline:
  dataset: !TabularDataset.from_path
    train_path: tests/data/dummy_tabular/train.csv
    val_path: tests/data/dummy_tabular/val.csv
    sep: ','
    transform:
      text: !TextField
      label: !LabelField
  model: !TextClassifier
    embedder: !Embedder
      embedding: !torch.Embedding
        num_embeddings: !@ dataset.text.vocab_size
        embedding_dim: 30
      encoder: !PooledRNNEncoder
        input_size: 30
        rnn_type: lstm
        n_layers: 1
        hidden_size: 16
    output_layer: !SoftmaxLayer
      input_size: !@ model[embedder].encoder.rnn.hidden_size
      output_size: !@ dataset.label.vocab_size

  exporter: !Exporter
    model: !@ model
    text: !@ dataset.text
"""

        exp = exp.format(d)
        f.write(exp)
        f.flush()
        ret = subprocess.run(['flambe', f.name, '-i'])
        assert ret.returncode == 0

        # Then run a builder

        builder = """
flambe_inference: tests/data/dummy_extensions/inference/
---

!Builder

destination: {0}

component: !flambe_inference.DummyInferenceEngine
  model: !TextClassifier.load_from_path
    path: {1}
"""
        base = os.path.join(d, "output__exporter", "exporter")
        path_aux = [x for x in os.listdir(base) if os.path.isdir(os.path.join(base, x))][0]  # Should be only 1 folder bc of no variants
        model_path = os.path.join(base, path_aux, "checkpoint", "checkpoint.flambe", "model")

        builder = builder.format(d2, model_path)
        f2.write(builder)
        f2.flush()

        ret = subprocess.run(['flambe', f2.name, '-i'])
        assert ret.returncode == 0

        # The extensions needs to be imported using extensions.py module
        extensions.import_modules(["flambe_inference"])

        # Import the module after import_modules (which registered tags already)
        from flambe_inference import DummyInferenceEngine

        eng1 = flambe.load(d2)

        assert type(eng1) is DummyInferenceEngine
        assert type(eng1.model) is TextClassifier

        extension_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests/data/dummy_extensions/inference")
        assert eng1._extensions == {"flambe_inference": extension_path}

        eng2 = DummyInferenceEngine.load_from_path(d2)

        assert type(eng2) is DummyInferenceEngine
        assert type(eng2.model) is TextClassifier

        assert eng2._extensions == {"flambe_inference": extension_path}

        assert module_equals(eng1.model, eng2.model)

