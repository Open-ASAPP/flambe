.. raw:: html

    <p align="center">
       <img src="imgs/Flambe_Logo_CMYK_FullColor.png" width="60%" align="middle">
    </p>

|

------------

|

.. image:: https://github.com/asappresearch/flambe/workflows/Run%20fast%20tests/badge.svg
    :target: https://github.com/asappresearch/flambe/actions
    :alt: Fast tests

.. image:: https://github.com/asappresearch/flambe/workflows/Run%20slow%20tests/badge.svg
    :target: https://github.com/asappresearch/flambe/actions
    :alt: Slow tests

.. image:: https://readthedocs.org/projects/flambe/badge/?version=latest
    :target: https://flambe.ai/en/latest/?badge=latest
    :alt: Documentation Status

|

Welcome to Flambé, a `PyTorch <https://pytorch.org/>`_-based library that allows users to:

* Run complex experiments with **multiple training and processing stages**
* **Search over hyperparameters**, and select the best trials
* Run experiments **remotely** over many workers, including full AWS integration
* Easily share experiment configurations, results, and model weights with others

Installation
------------

**From** ``PIP``:

.. code-block:: bash

    pip install flambe

**From source**:

.. code-block:: bash

    git clone git@github.com:asappresearch/flambe.git
    cd flambe
    pip install .


Getting started
---------------

Define an ``Experiment``:

.. code-block:: yaml

    !Experiment

    name: sst-text-classification

    pipeline:

      # stage 0 - Load the Stanford Sentiment Treebank dataset and run preprocessing
      dataset: !SSTDataset # this is a simple Python object, and the arguments to build it
        transform: # these arguments are passed to the init method
          text: !TextField
          label: !LabelField

      # Stage 1 - Define a model
      model: !TextClassifier
          embedder: !Embedder
            embedding: !torch.Embedding  # automatically use pytorch classes
              num_embeddings: !@ dataset.text.vocab_size # link to other components, and attributes
              embedding_dim: 300
            embedding_dropout: 0.3
            encoder: !PooledRNNEncoder
              input_size: 300
              n_layers: !g [2, 3, 4] # grid search over any parameters
              hidden_size: 128
              rnn_type: sru
              dropout: 0.3
          output_layer: !SoftmaxLayer
              input_size: !@ model[embedder][encoder].rnn.hidden_size # also use inner-links
              output_size: !@ dataset.label.vocab_size

      # Stage 2 - Train the model on the dataset
      train: !Trainer
        dataset: !@ dataset
        model: !@ model
        train_sampler: !BaseSampler
        val_sampler: !BaseSampler
        loss_fn: !torch.NLLLoss
        metric_fn: !Accuracy
        optimizer: !torch.Adam
          params: !@ train[model].trainable_params
        max_steps: 100
        iter_per_step: 100

      # Stage 3 - Eval on the test set
      eval: !Evaluator
        dataset: !@ dataset
        model: !@ train.model
        metric_fn: !Accuracy
        eval_sampler: !BaseSampler

    # Define how to schedule variants
    schedulers:
      train: !ray.HyperBandScheduler

All objects in the ``pipeline`` are subclasses of ``Component``, which
are automatically registered to be used with YAML. Custom ``Component``
implementations must implement ``run`` to add custom behavior when being executed.

Now just execute:

.. code-block:: bash

    flambe example.yaml

Note that defining objects like model and dataset ahead of time is optional; it's useful if you want to reference the same model architecture multiple times later in the pipeline.

Progress can be monitored via the Report Site (with full integration with Tensorboard):

.. raw:: html

    <p align="center">
       <kbd><img src="docs/image/report-site/partial.png" width="120%" align="middle" border="5"></kbd>
    </p>


Features
--------

* **Native support for hyperparameter search**: using search tags (see ``!g`` in the example) users can define multi variant pipelines. More advanced search algorithms will be available in a coming release!
* **Remote and distributed experiments**: users can submit ``Experiments`` to ``Clusters`` which will execute in a distributed way. Full ``AWS`` integration is supported.
* **Visualize all your metrics and meaningful data using Tensorboard**: log scalars, histograms, images, hparams and much more.
* **Add custom code and objects to your pipelines**: extend flambé functionality using our easy-to-use *extensions* mechanism.
* **Modularity with hierarchical serialization**: save different components from pipelines and load them safely anywhere.

Next Steps
-----------

Full documentation, tutorials and much more in https://flambe.ai

Contact
-------
You can reach us at flambe@asapp.com
