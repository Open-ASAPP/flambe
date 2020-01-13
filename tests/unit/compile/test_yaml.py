import pytest
from ruamel.yaml.compat import StringIO

from flambe.compile.yaml import load_config, dump_config
from flambe.compile.schema import Schema
from flambe.compile.registered_types import Registrable
from flambe.metric import AUC


@pytest.fixture
def make_classes():

    class A(Registrable):

        def __init__(self, akw1=0, akw2=None):
            self.akw1 = akw1
            self.akw2 = akw2

        @classmethod
        def from_yaml(cls, constructor, node, factory_name, tag):
            kwargs, = list(constructor.construct_yaml_map(node))
            return cls(**kwargs)

        @classmethod
        def to_yaml(cls, representer, node, tag):
            return representer.represent_mapping(tag, {"akw1": node.akw1, "akw2": node.akw2})

    class B(Registrable):

        def __init__(self, bkw1=0, bkw2=''):
            self.bkw1 = bkw1
            self.bkw2 = bkw2

        @classmethod
        def from_yaml(cls, constructor, node, factory_name, tag):
            kwargs, = list(constructor.construct_yaml_map(node))
            return cls(**kwargs)

        @classmethod
        def to_yaml(cls, representer, node, tag):
            return representer.represent_mapping(tag, {"bkw1": node.bkw1, "bkw2": node.bkw2})

    return A, B


class TestLoadConfig:

    def test_load_config(self):
        config = """!AUC"""
        x = load_config(config)
        assert isinstance(x, Schema)
        x = x()
        assert isinstance(x, AUC)

    def test_registrable_load_basic(self, make_classes):
        A, B = make_classes

        txt = """a: !A
  akw1: 8
  akw2: !B
    bkw1: 2
    bkw2: hello world
    """
        config = load_config(txt)
        a = config['a']
        assert a.akw1 == 8
        assert a.akw2 is not None
        assert hasattr(a.akw2, "bkw1")
        assert a.akw2.bkw1 == 2


class TestDumpConfig:

    def test_registrable_dump_basic(self, make_classes):
        A, B = make_classes

        txt = """!A
akw1: 8
akw2: !B
  bkw1: 2
  bkw2: hello world
    """

        b = B(2, "hello world")
        a = A(8, b)
        with StringIO() as s:
            dump_config(a, s)
            assert s.getvalue() == txt


class TestConfigRoundtrip:

    def test_registrable_roundtrip(self, make_classes):
        A, B = make_classes

        txt = """a: !A
  akw1: 8
  akw2: !B
    bkw1: 2
    bkw2: hello world
    """
        config = load_config(txt)
        with StringIO() as s:
            dump_config(config, s)
            assert s.getvalue() == txt
