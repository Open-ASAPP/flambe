==============================================================
Writing a multistage pipeline: BERT Fine-tuning + Distillation
==============================================================

It is common to want to train a model on a particular task, and reuse that model
or part of the model in a fine-tuning stage on a different dataset. Flambé allows
users to link directly to objects in a previous stage in the pipeline without
having to run two different experiments (more information on linking
:ref:`here <understanding-links_label>`)

For this tutorial, we look at a recent use case in natural language processing,
namely fine-tuning a `BERT <https://github.com/google-research/bert>`_
model on a text classification task, and applying knowledge
distillation on that model in order to obtain a smaller model of high performance.
Knowledge distillation is interesting as BERT is relativly slow, which can hinder
its use in production systems.


First step: BERT fine-tuning
----------------------------

We start by taking a pretrained BERT encoder, and we fine-tune it on the :class:`~flambe.nlp.classification.SSTDataset`
by adding a linear output layer on top of the encoder. We start with the dataset, and
apply a special TextField object which can load the pretrained vocabulary learned by
BERT.

The :class:`~flambe.nlp.classification.SSTDataset` below inherits from our :class:`~flambe.dataset.TabularDataset`
component. This object takes as input a ``transform`` dictionary, where you can specify
:class:`~flambe.field.Field` objects. A :class:`~flambe.field.Field`
is considered a featurizer: it can take an arbitrary number of columns and return an
any number of features.

.. tip:: You are free to completely override the :class:`~flambe.dataset.Dataset`
          object and not use :class:`~flambe.field.Field`, as long as you follow its interface:
          :class:`~flambe.dataset.Dataset`.


In this example, we apply a :class:`~flambe.nlp.transformers.bert.BertTextField`
and a :class:`~flambe.field.LabelField`.


.. code-block:: yaml

    dataset: !SSTDataset
        transform:
            text: !BertTextField
                alias: 'bert-base-uncased'
            label: !LabelField

.. tip::
    By default, fields are aligned with the input columns, but one can also make an explicit
    mapping if more than one feature should be created from the same column:

    .. code-block:: yaml

        transform:
            text:
                columns: 0
                field: !BertTextField
                    alias: 'bert-base-uncased'
            label:
                columns: 1
                field: !LabelField

Next we define our model. We use the :class:`~flambe.nlp.classification.TextClassifier`
object, which takes an :class:`~flambe.nn.Embedder`, and an output layer. Here,
we use the :class:`~flambe.nlp.transformer.BertEmbedder` 

.. code-block:: yaml

    teacher: !TextClassifier

      embedder: !BertEmbedder
        pool: True

      output_layer: !SoftmaxLayer
        input_size: !@ model.embedder.hidden_size
        output_size: !@ dataset.label.vocab_size  # We link the to size of the label space

Finally we put all of this in a :class:`~flambe.learn.Trainer` object, which will execute training.

.. tip:: Any component can be specified at the top level in the pipeline or be an argument
        to another :class:`~flambe.compile.Component` objects. A :class:`~flambe.compile.Component`
        has a run method which for many objects consists of just
        a ``pass`` statement, meaning that using them at the top level is equivalent to declaring them.
        The :class:`~flambe.learn.Trainer`
        however executes training through its run method, and will therefore be both declared and executed.

  finetune: !Trainer
    dataset: !@ dataset
    train_sampler: !BaseSampler
      batch_size: 16
    val_sampler: !BaseSampler
      batch_size: 16
    model: !@ teacher
    loss_fn: !torch.NLLLoss
    metric_fn: !Accuracy
    optimizer: !AdamW
      params: !@ finetune.model.trainable_params
      lr: 0.00005


Second step: Knowledge distillation
-----------------------------------

We now introduce a second model, which we will call the student model:


.. code-block:: yaml

    student: !TextClassifier

      embedder: !Embedder
        embedding: !Embeddings
          num_embeddings: !@dataset.text.vocab_size
          embedding_dim: 300
        encoder: !PooledRNNEncoder
          input_size: 300
          rnn_type: sru
          n_layers: 2
          hidden_size: 256
        pooling: !LastPooling
      output_layer: !SoftmaxLayer
        input_size: !@ student.embedder.encoder.hidden_size
        output_size: !@ dataset.label.vocab_size

.. attention::
    Note how this new model is way less complex than the original layer, being more appropriate
    for productions systems.

In the above example, we decided to reuse the same embedding layer, which
allows us not to have to provide a new :class:`~flambe.field.Field` to the dataset. However, you
may also decide to perform different preprocessing for the student model:

.. code-block:: yaml

    dataset: !SSTDataset
        transform:
            teacher_text: !BERTTextField.from_alias
                alias: 'bert-base-uncased'
                lower: true
            label: !LabelField
            student_text: !TextField

We can now proceed to the final step of our pipeline which is the :class:`~flambe.learn.distillation.DistillationTrainer`.
The key here is to link to the teacher model that was obtained in the ``finetune`` stage above.

.. tip::
    You can specify to the :class:`~flambe.learn.distillation.DistillationTrainer` which columns of the dataset
    to pass to the teacher model, and which to pass to the student model through the
    ``teacher_columns`` and ``student_columns`` arguments.


.. code-block:: yaml

    distill: !DistillationTrainer
      dataset: !@ dataset
      train_sampler: !BaseSampler
        batch_size: 16
      val_sampler: !BaseSampler
        batch_size: 16
      teacher_model: !@ finetune.model
      student_model: !@ student
      loss_fn: !torch.NLLLoss
      metric_fn: !Accuracy
      optimizer: !torch.Adam
        params: !@ distill.student_model.trainable_params
        lr: 0.00005
      alpha_kl: 0.5
      temperature: 1

.. attention::
    Linking to the teacher model directly would use the model pre-finetuning, so we link to
    the model inside the ``finetune`` stage. Note that for these links to work, it's important
    for the :class:`~flambe.learn.Trainer` object to have the ``model`` as instance attribute.

That's it! You can find the full configuration below.


Full configuration
------------------


.. code-block:: yaml

  !Experiment

  name: fine-tune-bert-then-distill
  pipeline:

    dataset: !SSTDataset
        transform:
            text: !BertTextField
                alias: 'bert-base-uncased'
            label: !LabelField

    teacher: !TextClassifier
      embedder: !BertEmbedder
        pool: True
      output_layer: !SoftmaxLayer
        input_size: !@ teacher.embedder.hidden_size
        output_size: !@ dataset.label.vocab_size  # We link the to size of the label space

    student: !TextClassifier
      embedder: !Embedder
        embedding: !Embeddings
          num_embeddings: !@ dataset.text.vocab_size
          embedding_dim: 300
        encoder: !PooledRNNEncoder
          input_size: 300
          rnn_type: sru
          n_layers: 2
          hidden_size: 256
        pooling: last
      output_layer: !SoftmaxLayer
        input_size: !@ student.embedder.encoder.hidden_size
        output_size: !@ dataset.label.vocab_size

    finetune: !Trainer
      dataset: !@ dataset
      train_sampler: !BaseSampler
        batch_size: 16
      val_sampler: !BaseSampler
        batch_size: 16
      model: !@ teacher
      loss_fn: !torch.NLLLoss
      metric_fn: !Accuracy
      optimizer: !AdamW
        params: !@ finetune.model.trainable_params
        lr: 0.00005

    distill: !DistillationTrainer
      dataset: !@ dataset
      train_sampler: !BaseSampler
        batch_size: 16
      val_sampler: !BaseSampler
        batch_size: 16
      teacher_model: !@ finetune.model
      student_model: !@ student
      loss_fn: !torch.NLLLoss
      metric_fn: !Accuracy
      optimizer: !torch.Adam
        params: !@ distill.student_model.trainable_params
        lr: 0.00005
      alpha_kl: 0.5
      temperature: 1
