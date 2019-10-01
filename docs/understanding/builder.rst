========
Builders
========

A :class:`~flambe.export.Builder` is a simple :class:`~flambe.runnable.Runnable` that can be used to create
any :class:`~flambe.compile.Component` post- :class:`~flambe.experiment.Experiment`, and export it to a local or remote location.

``Builders`` decouple the inference logic with the experimentation logic, allowing users
to iterate through inference contracts independently without needing to rerun an
:class:`~flambe.experiment.Experiment`.

.. hint::
 A :class:`~flambe.export.Builder` should be used to build inference engines that rely
 on previous experiments' artifacts.

Motivation
----------

Let's assume that a user wants to train a binary classifier using an :class:`~flambe.experiment.Experiment`:

.. code-block:: yaml

    ext: /path/to/my/extensions
    ---
    !Experiment
    ..
    pipeline:
        ...
        model: !ext.MyBinaryClassifier


Now, the user needs to implement an inference object ``ClassifierEngine`` that has a
method ``predict`` that performs the forward pass on the trained ``model``:

.. code-block:: python
    :linenos:

    from flambe.nn import Module
    from flambe.compile import Component

    class ClassifierEngine(Component):

       def __init__(self, model: Module):
          self.model = model

       def predict(self, **kwargs):
          # Custom code (for example, requests to APIs)
          p = self.model(feature)
          return {"POSITIVE": p, "NEGATIVE": 1-p}

By implementing :class:`~flambe.compile.Component`, the user can use a :class:`~flambe.export.Builder` to build this object:

.. code-block:: yaml

    ext: /path/to/my/extensions
    ---
    !Builder
    
    storage: s3
    destination: my-bucket

    ..
    component: !ClassifierEngine
        ...
        model: !ext.MyBinaryClassifier.load_from_path:
          path: /path/to/saved/modeel

The inference object will be saved in ``s3://my-bucket``. Then the user can:

.. code-block:: python
    :linenos:

    import flambe

    inference_engine = flambe.load("s3://my-bucket")
    inference_engine.predict(...)
    # >> {"POSITIVE": 0.9, "NEGATIVE": 0.1}

.. important::
    Note that the inference logic is decoupled from the :class:`~flambe.experiment.Experiment`. If in the
    future the inference logic changes, there is no need of rerunning it.

.. note::
    **Why not just implement a plain Python class and use** :meth:`flambe.compile.serialization.load` **to get the model?**
    Because of being a :class:`~flambe.compile.Component`, this object will have all the features
    :class:`~flambe.compile.Component` has (YAML serialization, versioning,
    compatibility with other :class:`~flambe.runnable.Runnable` implementations, among others).

How to use a builder
--------------------

Usage is really simple. The most important parameters for a :class:`~flambe.export.Builder` are
the :class:`~flambe.compile.Component` and the destination:


.. code-block:: yaml

    !Builder
    
    storage: [ local | s3 ]
    destination: path/to/location

    ..
    component: !MyComponent
        params1: value1
        params2: value2
        ...
        paramsN: valueN



.. important::
    For a full list of parameters, go to :class:`~flambe.export.Builder`.


.. hint::
    If storage is **"s3"**, then the destination can be an S3 bucket folder. Flambé will
    take care of uploading the built artifacts.

Future Work
-----------

The goal is to develop builders for different technologies. For example, a ``DockerBuilder`` that is able
to build a Docker container based on a :class:`~flambe.compile.Component`.
