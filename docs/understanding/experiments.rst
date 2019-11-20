.. _understanding-experiments_label:

===========
Experiments
===========

The top level object in every configuration file must be a :class:`~flambe.runnable.Runnable`, the most
common and useful being the :class:`~flambe.experiment.Experiment` class which facilitates
executing a ML pipeline.

The ``Experiment``'s most important parameter is the :attr:`~flambe.experiment.Experiment.pipeline`, where users can define
a `DAG <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_ of ``Components`` describing how dataset, models, training procedures, etc interact
between them.

.. attention::
    For a full specification of ``Experiment``, see :class:`~flambe.experiment.Experiment`

The implementation of :class:`~flambe.experiment.Experiment` and its `pipeline`` uses
`Ray's Tune <https://ray.readthedocs.io/en/latest/tune.html>`_ under the hood.

Pipeline
--------

A ``pipeline`` is defined as a list of ``Components`` that will be executed **sequentially**.
Each :class:`~flambe.compile.Component`
is identified by a **key that can be used for later linking**.

Let's assume that we want to define an experiment that consists on:

1. Pick dataset A and preprocess it.
2. Train model A on dataset A.
3. Preprocess dataset B.
4. Finetune the model trained in 2. on dataset B.
5. Evaluate the fine tuned model on dataset A testset.

All these stages can be represented by a sequential ``pipeline`` in a simple
and readable way:

.. code-block:: yaml

    pipeline:
        dataset_A: !SomeDataset
          ...

        model_A: !SomeModel
           ...

        trainer: !Trainer
           model: !@ model_A
           dataset: !@ dataset_A
           ...

        dataset_B: !Trainer
           ...

        fine_tunning: !Trainer
           model: !@ trainer.model
           dataset: !@ dataset_B
           ...

        eval: !Evaluator
           model: !@ trainer.model
           dataset: !@ dataset_A

Note how this represents a DAG where the nodes are the ``Components`` and the edges
are the links to attributes of previouly defined ``Components``.

.. _understanding-links_label:

Linking
-------

As seen before in :ref:`starting-quickstart_label`, stagegs in the ``pipeline`` are
connected using ``Links``.

Links can be used anywhere in the ``pipeline`` to refer to earlier components or any of their attributes.

During the compilation that is described above in :ref:`understanding-component-delayed-init_label`
we actually resolve the links to their intended value, but cache the original link representation
so that we can dump back to YAML with the original links later.

.. _understanding-search-options_label:

Search Options
--------------

``Experiment`` supports declaring multiple variants in the ``pipeline`` by making use of
the search tags:

.. code-block:: yaml

    !Experiment
    ...
    pipeline:
        ...
        model: !TextClassifier
           ...
           n_layers: !g [2, 3, 4]

        ...

The value ``!g [2, 3, 4]`` indicates that each of the
values should be tried. Flambé will create internally 3 variants of the model.

**You can specify grid search options search for any parameter in your config,
without changing your code to accept a new type of input! (in this case**
``n_layers`` **still receives an** ``int`` **)**


.. tip::
    You can also search over ``Components`` or even links:

    .. code-block:: yaml

        !Experiment
        ...

        pipeline:
          dataset: !SomeDataset
            transform:
              text: !g
              - !SomeTextField {{}}  # Double braces needed here
              - !SomeOtherTextField {{}}

**Types of search options**

``!g``
    Previously shown. It grids over all its values

    .. code-block:: yaml

        param: !g [1, 2, 3]  # grids over 1, 2 and 3.
        param: !g [0.001, 0.01]  # grids over 0.001 and 0.01

``!s``
    Yields k values from a range (low, high). If both ``low`` and ``high`` are int
    values, then ``!s`` will yield int values. Otherwise, it will yield float values.

    .. code-block:: yaml

        param: !s [1, 10, 5]  # yiels 5 int values from 1 to 10
        param: !s [1.5, 2.2, 5]  # yiels 5 float values from 1.5 to 2.2
        param: !s [1.5, 2.2, 5, 2]  # yiels 5 float values from 1.5 to 2.2, rounded to 2 decimals


**Combining Search tags**

Search over different attributes at the same time will have a combinatorial effect.

For example:

.. code-block:: yaml

    !Experiment
    ...
    pipeline:
        ...
        model: !TextClassifier
           ...
           n_layers: !g [2, 3, 4]
           hidden_size: !g [128, 256]

This will produce 6 variants (3 ``n_layers`` values times 2 ``hidden_size`` values)

**Variants inheritance**

.. attention::
   **Any object that links to an attribute of an object that describes multiple variants
   will inherit those variants.**

    .. code-block:: yaml

        !Experiment
        ...
        pipeline:
            ...
            model: !TextClassifier
               n_layers: !g [2, 3, 4]
               hidden_size: !g [128, 256]
               ...
            trainer: !Trainer
               model: !@ model
               lr: !g [0.01, 0.001]
               ...

            evaluator: !Evaluator
               model: !@ trainer.model

  The ``trainer`` will have 12 variants (6 from ``model`` times 2 for the ``lr``).
  ``eval`` will run for 12 variants as it links to ``trainer``.


Reducing
--------

``Experiment`` provides a :attr:`~flambe.experiment.Experiment.reduce` mechanism so that variants don't flow down the ``pipeline``.
**reduce** is declared at the ``Experiment`` level and it can specify the number of variants to reduce
to for each ``Component``.

.. code-block:: yaml

    !Experiment
    ...
    pipeline:
        ...
        model: !TextClassifier
           n_layers: !g [2, 3, 4]
           hidden_size: !g [128, 256]
        trainer: !Trainer
           model: !@ model
           lr: !g [0.01, 0.001]

        evaluator: !Evaluator
           ...
           model: !@ trainer.model

     reduce:
       trainer: 2

Flambé will then pick **the best 2 variants before finishing executing ``trainer``**. This means
``eval`` will receive the best 2 variants only.

Resources (Additional Files and Folders)
----------------------------------------

The :attr:`~flambe.experiment.Experiment.resources` argument lets users specify files that can be used in the
:class:`~flambe.experiment.Experiment` (usually local datasets, embeddings or other files).

For example:

.. code-block:: yaml

    !Experiment
    ...

    resources:
        data: path/to/train.csv
        embeddings: s3://mybucket/embeddings.bin
    ...

In case a resource is a remote URL, then flambé will download the file fow you (relying on the user local permissions)

.. attention:: Currently S3 and HTTP hosted resources are supported.

``resources`` can be referenced in the pipeline via linking:

.. code-block:: yaml

    !Experiment
    ...

    resources:
        ...
        embeddings: path/to/embeddings.txt

    pipeline:
        ...
          some_field: !@ embeddings


**Resources in remote experiment**

When running remote experiments, all resources will be rsynced into the instances so that they are available in
the cluster **unless a ``!cluster`` tag is specified**.

The ``!cluster`` tag is useful when the cluster needs to handle the resources. The local process will just
ignore those tagged resources.

For example:

.. code-block:: yaml

    !Experiment
    ...

    resources:
        data: !cluster path/to/train.csv  # This file is already in all instances of the cluster
    ...
 
When running this example in a cluster, then no ``rsync`` will be involved as flambé assumes the resource
path ``path/to/train.csv`` exists in all instances of the cluster.

.. tip::
    You can also specify remote URL with the ``!cluster`` tag:

    .. code-block:: yaml

        !Experiment
        ...

        resources:
            data: !cluster s3://bucket/data.csv
        ...

    In this case the cluster will download the data instead of the local process (if it has permissions to
    do so)


.. attention:: The ``!cluster`` tag is only useful in remote experiments. If the user is running local experiments, using ``!cluster`` will fail.

.. _understanding-experiments-scheduling_label:

Scheduling and Reducing Strategies
----------------------------------

When running a search over hyperparameters, you may want to run a more
sophisticated scheduler. Using `Tune <https://ray.readthedocs.io/en/latest/tune.html>`_,
you can already use algorithms such as
HyperBand, and soon more complex search algorithms like HyperOpt will be available.

.. code-block:: yaml

    schedulers:
        b1: !ray.HyperBandScheduler

    pipeline:
        b0: !ext.TCProcessor
            dataset: !ext.SSTDataset
        b1: !Trainer
            train_sampler: !BatchSampler
                data: !@ b0.train
                batch_size: !g [32, 64, 128]
            model: ...
        b2: !Evaluator
            model: !@ b1.model

General Logging
----------------

We adopted the standard library's `logging <https://docs.python.org/3/howto/logging.html>`_
module for logging:

.. code-block:: python
    :linenos:

    import logging
    logger = logging.getLogger(__name__)
    ...
    logger.info("Some info here")
    ...
    logger.error("Something went wrong here...")

The best part of the logging paradigm is that you can instantly start logging
in any file in your code without passing any data or arguments through your
object hierarchy.

.. important::
    By default, only log statements at or above the ``INFO`` log level will be shown
    in the console. The rest of the logs will be saved in ``~/.flambe/logs`` (more on this
    in :ref:`advanced-debugging_label`)

In order to show all logs in the console, you can use the ``--vebose`` flag
when running flambé:

.. code-block:: bash

    flambe my_config_file.yaml --verbose

Tensorboard Logging
-------------------

Flambé provides full integration with `Tensorboard <https://www.tensorflow.org/guide/summaries_and_tensorboard>`_.
Users can easily have data routed to Tensorboard through the logging
interface:

.. code-block:: python
    :linenos:

    from flambe import log
    ...
    loss = ... # some calculation here
    log('train loss', loss, step)

Where the first parameter is the tag which Tensorboard uses to name the value.
The logging system will automatically detect the type and make sure it goes to the right Tensorboard function.
See :func:`flambe.logging.log` in the package reference.

Flambé provides also logging special types of data:

* :func:`flambe.logging.log_image` for images
* :func:`flambe.logging.log_histogram` for distributions and histograms
* :func:`flambe.logging.log_pr_curves` for displaying PR curves
* :func:`flambe.logging.log_text` for displaying text

See the :mod:`~flambe.logging` for more information on how to use this logging methods.

Script Usage
------------

If you're using the :class:`flambe.learn.Script` object to wrap an existing piece
of code with a command-line based interface, all of the logging information above
still applies to you!

See more on Scripts in :ref:`tutorials-script_label`.

Checkpointint and Saving
------------------------

As :ref:`starting-quickstart_label` explains, flambé saves an :class:`~flambe.experiment.Experiment` in
a hierarchical way so that ``Components`` can be accessed independant to each other.
Specifically, our save files are a directory by default, and
include information about the class name, version, source code, and YAML config,
in addition to the state that PyTorch normally saves, and any custom state
that the implementer of the class may have included.

For example, if you initialize and use the following object as a part of your ``Experiment``:

.. code-block:: yaml

    !TextClassifier
    embedder: !Embedder
      embedding: !torch.Embedding
        input_size: !@ b0.text.vocab_size
        embedding_size: 300
      encoder: !PooledRNNEncoder
        input_size: 300
        rnn_type: lstm
        n_layers: 2
        hidden_size: 256
    output_layer: !SoftmaxLayer
      input_size: !@ b1[model][encoder][encoder].rnn.hidden_size
      output_size: !@ b0.label.vocab_size

Then the save directory would look like the following:

::

    save_path
    ├── state.pt
    ├── config.yaml
    ├── version.txt
    ├── source.py
    ├── embedder
    │   ├── state.pt
    │   ├── config.yaml
    │   ├── version.txt
    │   ├── source.py
    │   ├── embedding
    │   │   ├── state.pt
    │   │   ├── config.yaml
    │   │   ├── version.txt
    │   │   └── source.py
    │   └── encoder
    │       ├── state.pt
    │       ├── config.yaml
    │       ├── version.txt
    │       └── source.py
    └── output_layer
        ├── state.pt
        ├── config.yaml
        ├── version.txt
        └── source.py

Note that each subdirectory is self-contained: if it's possible to load that object
on its own, you can load from just that subdirectory.

.. important::
  As seen before, each variant of a :class:`~flambe.compile.Component` will have it's separate output folder.

.. note::
  Flambé will save in this format automatically after each ``Component`` of the pipeline executes
  :meth:`~flambe.runnable.Runnable.run`. As there are objects that execute :meth:`~flambe.runnable.Runnable.run`
  multiple times (for example, :class:`~flambe.learn.Trainer`),
  each time the state will be overriden by the latest one (checkpointing).

Resuming
--------

:class:`~/flambe.experiment.Experiment` has a way of resuming perviously run experiments:

.. code-block:: yaml

    !Experiment
    resume: trainer
    ...
    pipeline:
        ...
        model: !TextClassifier
           ...
           n_layers: !g [2, 3, 4]
           hidden_size: !g [128, 256]

        trainer: !Trainer
           ...
           model: !@ model
           lr: !g [0.01, 0.001]

        other_trainer: !Trainer
           ...
           model: !@ trainer.model

By providing a ``Component`` keyname (or a list of them) that belong to the ``pipeline``, then
**flambé will resume AFTER all the given blocks, i.e. it would not execute those blocks and continue the experiment after them.**

Debugging
---------

``Experiment`` has a debugging option that is only available in local executions (not remotely).
This is activated by adding ``debug: True`` at the top level of the YAML.

When debugging is on, a debugger will appear before executing ``run`` on each ``Component``.

.. warning::
    Debugging is not enabled when running remote experiments.

.. _understanding-experiments-custom-state_label:

Adding Custom State
-------------------

Users can add other data to
the state that is saved in the save directory. If you just want to have some
additional instance attributes added, you can register them at the end of the
``__init__`` method:

.. code-block:: python

    class MyModel(flambe.nn.Module):

        def __init__(self, x, ...):
            super().__init__(...)
            ...
            self.x = x,
            self.y = None
            self.register_attrs('x', 'y')

This will cause the ``get_state`` method to start including `x` and `y` in the
state dict for instances of ``MyModel``, and when you load state into instances
of ``MyModel`` it will know to update these attributes.

If you want more flexibility to manipulate the state_dict or add computed
properties you can override the :meth:`~flambe.compile.Component._state` and
:meth:`~flambe.compile.Component._load_state` methods.
