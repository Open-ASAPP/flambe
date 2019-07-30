.. raw:: html

    <p align="center">
       <img src="imgs/Flambe_Logo_CMYK_FullColor.png" width="60%" align="middle">
    </p>

|

.. image:: https://circleci.com/gh/Open-ASAPP/flambe.svg?style=svg
    :target: https://circleci.com/gh/Open-ASAPP/flambe

|

.. image:: https://readthedocs.org/projects/flambe/badge/?version=latest
    :target: https://flambe.ai/en/latest/?badge=latest
    :alt: Documentation Status

------------

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

    pip install flambe  # CPU Version
    # OR
    pip install flambe[cuda]  # With GPU / CUDA support
   
**From source**:

.. code-block:: bash

    git clone git@github.com:Open-ASAPP/flambe.git
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
      dataset: !SSTDataset
        transform:
          text: !TextField
          label: !LabelField

      # Stage 1 - Define a model
      model: !TextClassifier
          embedder: !Embedder
            embedding: !Embeddings
              num_embeddings: !@ dataset.text.vocab_size  # Link to previously defined attributes
              embedding_dim: 300
            encoder: !PooledRNNEncoder
              input_size: 300
              n_layers: !g [2, 3, 4]  # Grid search over hyperparameters!
              hidden_size: 256
              rnn_type: lstm
          output_layer: !SoftmaxLayer
              input_size: !@ model.embedder.encoder.rnn.hidden_size
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
          params: !@ train.model.trainable_params
        max_steps: 200
        iter_per_step: 500
        

      # Stage 3 - Eval on the test set
      eval: !Evaluator
        dataset: !@ dataset
        model: !@ train.model
        metric_fn: !Accuracy
        eval_sampler: !BaseSampler

    # Define how to schedule variants
    schedulers:
      train: !tune.HyperBandScheduler

    # Reduce to the best N variants (the best one in this case)
    reduce:
      train: 1

Now just execute:

.. code-block:: bash

    flambe example.yaml 

Note that defining objects like model and dataset ahead of time is optional; it's usefull if you want to reference the same model architecture multiple times later in the pipeline.

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

