import pytest
import tempfile
import sys

from flambe.runnable import SafeExecutionContext, error


@pytest.fixture
def context():
    t = tempfile.NamedTemporaryFile(mode="w+")
    def ex(config):
        t.write(config)
        t.flush()
        return SafeExecutionContext(t.name)
    yield ex
    t.close()


def test_preprocessor_non_existing_file():
    with pytest.raises(FileNotFoundError):
        ex = SafeExecutionContext("/non/existing/file")
        ex.preprocess()


def test_preprocessor_empty():
    with pytest.raises(TypeError):
        ex = SafeExecutionContext()
        ex.preprocess()


def test_preprocessor_none_file():
    with pytest.raises(TypeError):
        ex = SafeExecutionContext(None)
        ex.preprocess()


def clean_imported_pkg(name, added_paths):
    """Run this everytime something gets imported.
    It removes recursively all modules imported under name

    """
    for k in list(sys.modules.keys()):
        if name == k or k.startswith(name + '.'):
            del sys.modules[k]

    sys.path = [x for x in sys.path if x not in added_paths]


def test_preprocessor_not_yaml(context):
    config = """
Not a YAML file content
"""
    with pytest.raises(ValueError):
        ex = context(config)
        ex.preprocess()


def test_preprocessor_wrong_yaml(context):
    with pytest.raises(error.ParsingRunnableError):
        ex = SafeExecutionContext("tests/data/dummy_configs/wrong_config.yaml")
        ex.first_parse()


def test_preprocessor_none_content(context):
    config = """
!Experiment

name: some-name

resources:
pipeline:

"""
    with pytest.raises(error.ParsingRunnableError):
        ex = context(config)
        content, ext = ex.first_parse()
        runnable = ex.compile_runnable(content)
        runnable.parse()


def test_preprocessor_name_none(context):
    config = """
!Experiment

name:

resources:

pipeline:
    mod: !Module

"""
    with pytest.raises(error.ParsingRunnableError):
        ex = context(config)
        content, ext = ex.first_parse()
        runnable = ex.compile_runnable(content)
        runnable.parse()


def test_preprocessor_invalid_name(context):
    config = """
!Experiment

name: some-invalid-name-!@#$%^&&&&&^%$#@

pipeline:
    mod: !Module
"""
    with pytest.raises(error.ParsingRunnableError):
        ex = context(config)
        content, ext = ex.first_parse()
        runnable = ex.compile_runnable(content)
        runnable.parse()


def test_preprocessor_invalid_name2(context):
    config = """
!Experiment

name: /some/invalid/name

pipeline:
    mod: !Module
"""
    with pytest.raises(error.ParsingRunnableError):
        ex = context(config)
        content, ext = ex.first_parse()
        runnable = ex.compile_runnable(content)
        runnable.parse()



def test_preprocessor_valid_resources(context):
    config = """
!Experiment

name: random-name

resources:
  r0: !remote /remote/path

pipeline:
    mod: !Module
"""
    ex = context(config)
    content, ext = ex.first_parse()
    runnable = ex.compile_runnable(content)
    runnable.parse()


def test_preprocessor_valid_paths(context):
    config = """
!Experiment

name: random-name

pipeline:
  b0: !Something
    b1: !Other
      b2: something
"""
    ex = context(config)
    content, ext = ex.first_parse()
    runnable = ex.compile_runnable(content)
    runnable.parse()


def test_preprocessor_extensions_non_package(context):
    config = """
flambe_script: tests/data/dummy_extensions/script/setup.py
---
!Experiment

name: some-name

pipeline:
    mod: !Module
"""
    with pytest.raises(ImportError):
        ex = context(config)
        ex.preprocess(install_ext=True)


def test_preprocessor_extensions_invalid_mod_name(context):
    config = """
script: tests/data/dummy_extensions/script
---
!Experiment

name: some-name

pipeline:
    mod: !Module
"""
    with pytest.raises(ImportError):
        ex = context(config)
        ex.preprocess(install_ext=True)


def test_preprocessor_valid_extensions(context):
    config = """
flambe_script: tests/data/dummy_extensions/script
---
!Experiment

name: some-name

pipeline:
    mod: !Module
"""
    ex = context(config)
    ex.preprocess(install_ext=True)


def test_preprocessor_custom_experiment(context):
    config = """
flambe_runnable: tests/data/dummy_extensions/runnable
---
!flambe_runnable.DummyRunnable
"""
    ex = context(config)
    ex.preprocess(install_ext=True)


def test_preprocessor_custom_experiment_invalid(context):
    # Importing the wrong package
    config = """
flambe_script: tests/data/dummy_extensions/runnable
---
!flambe_runnable.DummyRunnable
"""
    with pytest.raises(AttributeError):
        ex = context(config)
        ex.preprocess(install_ext=True)


def test_preprocessor_unknown_tag(context):
    config = """
    !Experiment

    name: random-name

    pipeline:
        b0: !Something
    """
    with pytest.raises(error.TagError):
        context('').check_tags(config)


def test_preprocessor_unknown_factory(context):
    config = """
    !Experiment

    name: random-name

    pipeline:
        b0: !Trainer.something
    """
    with pytest.raises(error.TagError):
        context('').check_tags(config)
