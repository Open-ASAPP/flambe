.. _tutorials-script_label:

==============================
Converting a script to Flambé
==============================

In many situations, you may already have a script that performs your full training routine,
and you would like to avoid doing any intergation work to leverage some of the tools in Flambé,
namely launching variants of the script on a large cluster, and using the Flambé logging system.

For this particular use case, Flambé offers a ``Script`` component, which takes as input
an executable module (your script), and the relevant arguments. The script should read arguments
from ``sys.argv``, which means that traditional scripts, and scripts that use tools such as argparse
are supported.

.. attention:: The ``Script`` object only consists of a single step, and is therefore not
               compatible with checkpointing or trial schedulers such as Hyperband. It is however
               possible to use with hyperparameter search algorithms.


Wrapping your script in a pip installable
-----------------------------------------

Say you have the following directory structure for your project:

::

    my_project
    ├── model.py
    ├── processing.py
    └── train.py

Where ``train.py`` is your target script which uses ``argparse`` to read arguments.
The first step is to convert your project into a pip installable.

::

    my_pip_installable
    ├── setup.py
    └── my_project
        ├── __init__.py
        ├── model.py
        ├── processing.py
        └── train.py

Your setup should contain all of the external library requirements, used by your script.
Once this is done, you should make an attempt at running your script using the ``-m`` argument,
which will treat ``train.py`` as an executable module. You can do so by running:

.. code-block:: bash

    python -m my_project.train --arg1 value1 --arg2 value2

.. attention:: You will need to modify the imports in your script to use either relative imports or
               import from the top level package. In this example, this can be done by replacing
               ``import model`` by import ``.model``. Note that you only need to perform this change
               once and will still be able to run your script, normally, regardless of Flambé,
               using ``python -m``.

Writing a config file
---------------------

Once you have done the above step, you can use your script in Flambé as follows:


.. code-block:: yaml

    my_project: /path/to/my_pip_installable

    ---

    !Experiment
    
    name: example

    pipeline:
      stage_0: !Script
        script: my_project.train  # my_project is the name of the module
        args:
          arg1: !g [1, 5]  # Run a grid search over any arguments to your script
          arg2: 'foo'

That's it! You can now execute this configuration file, with the regular command:

.. code-block:: bash
    flambe config.yaml --cluster cluster.yaml

In order to see tensorboard logs, simply import the logger, and use it anywhere in your script:

.. code-block:: bash
    from flambe import log
