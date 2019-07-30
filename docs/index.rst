=================
Welcome to Flambé
=================

Welcome to Flambé, a `PyTorch <https://pytorch.org/>`_-based library that allows
users to:

* Run complex experiments with multiple training and processing stages.
* Search over an arbitrary number of parameters and reduce to the best trials.
* Run experiments remotely over many workers, including full AWS integration.
* Easily share experiment configurations, results and model weights with others.


**A simple Text Classification experiment**


.. code-block:: yaml

    !Experiment

    pipeline:

      # stage 0 - Load the Stanford Sentiment Treebank dataset and run preprocessing
      dataset: !SSTDataset
        transform:
          text: !TextField
          label: !LabelField

      # Stage 1 - Define a model
      model: !TextClassifier
          embedder: !Embedder
            embedding: !Embeddings
              num_embeddings: !@ dataset.text.vocab_size
              embedding_dim: 300
            encoder: !PooledRNNEncoder
              input_size: 300
              n_layers: !g [2, 3, 4]  # Grid search over hyperparameters!
          output_layer: !SoftmaxLayer
              input_size: !@ model.embedder.encoder.rnn.hidden_size
              output_size: !@ dataset.label.vocab_size

      # Stage 2 - Train the model on the dataset
      train: !Trainer
        dataset: !@ dataset
        model: !@ model
        train_sampler: !BaseSampler
           batch_size: 64
        model: !TextClassifier
        loss_fn: !torch.NLLLoss
        metric_fn: !Accuracy
        optimizer: !torch.Adam
          params: !@ train.model.trainable_params

      # Stage 3 - Eval on the test set
      eval: !Evaluator
        dataset: !@ dataset
        model: !@ train.model
        metric_fn: !Accuracy

    # Define how to schedule variants
    schedulers:
      train: !tune.HyperBandScheduler

    # Reduce to the best N variants
    reduce:
      train: 1
        ...

The experiment can be executed by running:

.. code:: bash

    flambe experiment.yaml

By defining a ``Cluster``:

.. code-block:: yaml

    !AWSCluster

    name: my-cluster  # Make sure to name your cluster

    factories_num: 2 # Number of factories to spin up, there is always just 1 orchestrator

    factories_type: g3.4xlarge
    orchestrator_type: t3.large

    key: '/path/to/ssh/key'
    
    ...

Then the same experiment can be run remotely:

.. code:: bash

    flambe experiment.yaml --cluster cluster.yaml

You can track the progress of your experiment using the Report Site and Tensorboard:

GIF HERE


**Getting Started**

Check out our :ref:`Installation Guide <starting-install_label>` and :ref:`starting-quickstart_label` sections to get up and
running with Flambé in just a few minutes!



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   starting/install
   starting/usage
   starting/motivation

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Understanding Flambé

   understanding/component
   understanding/runnables
   understanding/experiments
   understanding/report_site
   understanding/extensions
   understanding/clusters
   understanding/builder
   understanding/security
   understanding/advanced


.. toctree::
   :titlesonly:
   :hidden:
   :caption: Tutorials

   tutorials/script
   tutorials/custom
   tutorials/multistage

.. toctree::
   :titlesonly:
   :hidden:
   :caption: Package Reference

   autoapi/flambe/dataset/index
   autoapi/flambe/cluster/index
   autoapi/flambe/compile/index
   autoapi/flambe/dataset/index
   autoapi/flambe/experiment/index
   autoapi/flambe/export/index
   autoapi/flambe/field/index
   autoapi/flambe/learn/index
   autoapi/flambe/logging/index
   autoapi/flambe/metric/index
   autoapi/flambe/model/index
   autoapi/flambe/nlp/index
   autoapi/flambe/nn/index
   autoapi/flambe/runnable/index
   autoapi/flambe/runner/index
   autoapi/flambe/sampler/index
   autoapi/flambe/tokenizer/index
   autoapi/flambe/vision/index
