import pytest
import tempfile
from collections import abc, OrderedDict
import os
import pprint

import torch
import dill
import mock
from ruamel.yaml.compat import StringIO
from ruamel.yaml import YAML

# from flambe.compile import yaml
from flambe import Component, save_state_to_file, load_state_from_file, load, save
from flambe.compile import Registrable, yaml, make_component
from flambe.compile.serialization import _extract_prefix


FLAMBE_SOURCE_KEY = '_flambe_source'
FLAMBE_CLASS_KEY = '_flambe_class'
FLAMBE_CONFIG_KEY = '_flambe_config'
FLAMBE_DIRECTORIES_KEY = '_flambe_directories'
KEEP_VARS_KEY = 'keep_vars'
VERSION_KEY = '_flambe_version'


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


def check_mapping_equivalence(x, y, exclude_config=False):
    for key in x.keys():
        if key == KEEP_VARS_KEY or key == 'version':
            continue
        if key == FLAMBE_CONFIG_KEY and exclude_config:
            continue
        assert key in y
        if isinstance(x[key], abc.Mapping):
            check_mapping_equivalence(x[key], y[key], exclude_config=exclude_config)
        elif isinstance(x[key], torch.Tensor):
            assert isinstance(y[key], torch.Tensor)
            torch.equal(x[key], y[key])
        else:
            assert x[key] == y[key]


EXAMPLE_TRAINER_CONFIG = """
!Trainer
train_sampler: !BaseSampler
val_sampler: !BaseSampler
dataset: !TabularDataset
  train: [['']]
model: !RNNEncoder
  input_size: 300
  rnn_type: lstm
  n_layers: 2
  hidden_size: 256
loss_fn: !torch.NLLLoss
metric_fn: !Accuracy
optimizer: !torch.Adam
  params: []
max_steps: 2
iter_per_step: 2
"""


@pytest.fixture
def make_classes_2():

    class A(Component):

        def __init__(self, akw1=0, akw2=None):
            self.akw1 = akw1
            self.akw2 = akw2

    class B(Component):

        def __init__(self, bkw1=0, bkw2='', bkw3=99):
            self.bkw1 = bkw1
            self.bkw2 = bkw2
            self.bkw3 = bkw3

    class C(Component):

        def __init__(self, one, two):
            self.one = one
            self.two = two

    return A, B


class Basic(Component):
    pass


class Composite(Component):
    def __init__(self):
        self.leaf = Basic()


class BasicStateful(Component):
    def __init__(self):
        self.x = 2019
        self.register_attrs('x')
        # self.b = Basic()


class IntermediateTorch(Component, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf = Basic()


class IntermediateStatefulTorch(Component, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf = BasicStateful()
        self.linear = torch.nn.Linear(2, 2)


class RootTorch(Component):
    def __init__(self):
        super().__init__()
        self.model = IntermediateStatefulTorch()
        # self.linear = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 0.01)


class ComposableTorchStateful(Component, torch.nn.Module):
    def __init__(self, a: Component, b: int, c: torch.nn.Linear):
        super().__init__()
        self.child = a
        self.other_data = b
        self.linear = c
        self.register_attrs('other_data')


class ComposableTorchStatefulPrime(Component, torch.nn.Module):
    def __init__(self, a: Component, b: int, c: torch.nn.Linear):
        super().__init__()
        self.child = a
        self.other_data = b
        self.linear = c

    def _state(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'other_data'] = self.other_data
        return state_dict

    def _load_state(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        assert prefix + 'other_data' in state_dict
        self.other_data = state_dict[prefix + 'other_data']


class ComposableContainer(Component):
    def __init__(self, item: Component):
        self.item = item


def create_factory(class_):
    def _factory(from_config):
        if from_config:
            config = f"!{class_.__name__} {{}}\n"
            obj = yaml.load(config)()
            return obj
        else:
            obj = class_()
        return obj
    return _factory


@pytest.fixture
def basic_object():
    return create_factory(Basic)


@pytest.fixture
def nested_object():
    return create_factory(Composite)


@pytest.fixture
def basic_object_with_state():
    return create_factory(BasicStateful)


@pytest.fixture
def alternating_nn_module_with_state():
    return create_factory(RootTorch)


def schema_builder():
    config = """
!Basic
"""
    obj = yaml.load(config)
    return obj


def complex_builder(from_config):
    if from_config:
        config = """
!ComposableTorchStateful
a: !ComposableTorchStateful
  a: !ComposableTorchStateful
    a: !BasicStateful {}
    b: 2021
    c: !torch.Linear
      in_features: 2
      out_features: 2
  b: 2022
  c: !torch.Linear
    in_features: 2
    out_features: 2
b: 2023
c: !torch.Linear
  in_features: 2
  out_features: 2
"""
        obj = yaml.load(config)()
        return obj
    else:
        a1 = BasicStateful()
        b1 = 2021
        c1 = torch.nn.Linear(2, 2)
        a2 = ComposableTorchStateful(a1, b1, c1)
        b2 = 2022
        c2 = torch.nn.Linear(2, 2)
        a3 = ComposableTorchStateful(a2, b2, c2)
        b3 = 2023
        c3 = torch.nn.Linear(2, 2)
        obj = ComposableTorchStateful(a3, b3, c3)
        return obj


def complex_builder_nontorch_root(from_config):
    if from_config:
        config = """
!ComposableContainer
item:
  !ComposableTorchStatefulPrime
  a: !ComposableTorchStateful
    a: !ComposableTorchStateful
      a: !BasicStateful {}
      b: 2021
      c: !torch.Linear
        in_features: 2
        out_features: 2
    b: 2022
    c: !torch.Linear
      in_features: 2
      out_features: 2
  b: 2023
  c: !torch.Linear
    in_features: 2
    out_features: 2
"""
        obj = yaml.load(config)()
        return obj
    else:
        a1 = BasicStateful()
        b1 = 2021
        c1 = torch.nn.Linear(2, 2)
        a2 = ComposableTorchStateful(a1, b1, c1)
        b2 = 2022
        c2 = torch.nn.Linear(2, 2)
        a3 = ComposableTorchStateful(a2, b2, c2)
        b3 = 2023
        c3 = torch.nn.Linear(2, 2)
        item = ComposableTorchStateful(a3, b3, c3)
        obj = ComposableContainer(item)
        return obj


@pytest.fixture
def complex_multi_layered():
    return complex_builder


@pytest.fixture
def complex_multi_layered_nontorch_root():
    return complex_builder_nontorch_root

@pytest.fixture
def schema():
    return schema_builder

class TestHelpers:

    def test_extract_prefix(self):
        _extract_prefix


class TestState:

    def test_state_returns_not_None(self, basic_object):
        obj = basic_object(from_config=True)
        assert obj.get_state() is not None

    def test_state_metadata(self, basic_object):
        state = basic_object(from_config=False).get_state()
        assert hasattr(state, '_metadata')
        assert '' in state._metadata
        assert FLAMBE_DIRECTORIES_KEY in state._metadata
        assert FLAMBE_SOURCE_KEY in state._metadata['']
        assert VERSION_KEY in state._metadata['']
        assert state._metadata[''][FLAMBE_SOURCE_KEY] == "class Basic(Component):\n    pass\n"
        # assert state[FLAMBE_CONFIG_KEY] == ''
        assert '' in state._metadata[FLAMBE_DIRECTORIES_KEY] and len(state._metadata[FLAMBE_DIRECTORIES_KEY]) == 1
        assert state._metadata[''][VERSION_KEY] == '0.0.0'

    def test_state_config(self, basic_object):
        assert FLAMBE_CONFIG_KEY not in basic_object(from_config=False).get_state()._metadata['']
        obj = basic_object(from_config=True)
        state = obj.get_state()
        assert FLAMBE_CONFIG_KEY in state._metadata['']
        assert state._metadata[''][FLAMBE_CONFIG_KEY] == "!Basic {}\n"

    def test_state_nested_but_empty(self, nested_object):
        expected_state = {}
        expected_metadata = {'': {FLAMBE_SOURCE_KEY: "class Composite(Component):\n    def __init__(self):\n        self.leaf = Basic()\n", VERSION_KEY: "0.0.0", FLAMBE_CLASS_KEY: 'Composite'}, 'leaf': {FLAMBE_SOURCE_KEY: 'class Basic(Component):\n    pass\n', VERSION_KEY: '0.0.0', FLAMBE_CLASS_KEY: 'Basic'}, FLAMBE_DIRECTORIES_KEY: {'', 'leaf'}, KEEP_VARS_KEY: False}
        obj = nested_object(from_config=False)
        state = obj.get_state()
        assert state == expected_state
        check_mapping_equivalence(expected_metadata, state._metadata)
        check_mapping_equivalence(state._metadata, expected_metadata)

    def test_state_custom(self, basic_object_with_state):
        obj = basic_object_with_state(from_config=True)
        expected_state = {'x': 2019}
        assert obj.get_state() == expected_state

    # def test_state_custom_nested(nested_object_with_state):
    #     obj = nested_object_with_state()
    #     expected_state = {}
    #     assert obj.get_state() == expected_state
    #
    # def test_state_pytorch_empty(nn_modules):
    #     cls, cls_torch_first = nn_modules
    #     obj, obj_torch_first = cls(), cls_torch_first()
    #     expected_state = {}
    #     assert obj.get_state() == expected_state
    #     assert obj_torch_first.get_state() == expected_state
    #
    # def test_state_pytorch_nested_no_modules_no_parameters(nested_nn_module):
    #     obj = nested_nn_module()
    #     expected_state = {}
    #     assert obj.get_state() == expected_state
    #
    # def test_state_pytorch_alternating_nesting(alternating_nn_module):
    #     obj = alternating_nn_module()
    #     expected_state = {}
    #     assert obj.get_state() == expected_state

    def test_state_pytorch_alternating_nested_with_modules(self, alternating_nn_module_with_state):
        obj = alternating_nn_module_with_state(from_config=True)
        t1 = obj.model.linear.weight
        t2 = obj.model.linear.bias
        expected_state = {'model.leaf.x': 2019, 'model.linear.weight': t1, 'model.linear.bias': t2}
        root_source_code = dill.source.getsource(RootTorch)
        intermediate_source_code = dill.source.getsource(IntermediateStatefulTorch)
        leaf_source_code = dill.source.getsource(BasicStateful)
        expected_metadata = OrderedDict({FLAMBE_DIRECTORIES_KEY: set(['', 'model', 'model.leaf']), 'keep_vars': False, '': {VERSION_KEY: '0.0.0', FLAMBE_CLASS_KEY: 'RootTorch', FLAMBE_SOURCE_KEY: root_source_code, FLAMBE_CONFIG_KEY: "!RootTorch {}\n"},
                                                                                             'model': {VERSION_KEY: '0.0.0', FLAMBE_CLASS_KEY: 'IntermediateStatefulTorch', FLAMBE_SOURCE_KEY: intermediate_source_code, 'version': 1},  # TODO add config back: FLAMBE_CONFIG_KEY: "!IntermediateStatefulTorch {}\n"
                                                                                             'model.leaf': {VERSION_KEY: '0.0.0', FLAMBE_CLASS_KEY: 'BasicStateful', FLAMBE_SOURCE_KEY: leaf_source_code},  # TODO add config back: FLAMBE_CONFIG_KEY: "!BasicStateful {}\n"
                                                                                             'model.linear': {'version': 1}})
        state = obj.get_state()
        check_mapping_equivalence(state._metadata, expected_metadata)
        check_mapping_equivalence(expected_metadata, state._metadata)
        check_mapping_equivalence(state, expected_state)
        check_mapping_equivalence(expected_state, state)


class TestLoadState:

    def test_load_state_empty(self):
        pass

    def test_load_state_nested_empty(self):
        pass

    def test_load_state_custom_nested(self):
        pass

    def test_load_state_pytorch(self):
        pass

    def test_load_state_pytorch_alternating_nested(self):
        pass

    def test_state_complex_multilayered_nontorch_root(self, complex_multi_layered_nontorch_root):
        TORCH_TAG_PREFIX = "torch"
        make_component(torch.nn.Module, TORCH_TAG_PREFIX, only_module='torch.nn')

        obj = complex_multi_layered_nontorch_root(from_config=True)
        t1 = obj.item.child.linear.weight
        state = obj.get_state()
        new_obj = complex_multi_layered_nontorch_root(from_config=True)
        new_obj.load_state(state)
        t2 = new_obj.item.child.linear.weight
        assert t1.equal(t2)
        check_mapping_equivalence(new_obj.get_state(), obj.get_state())
        check_mapping_equivalence(obj.get_state(), new_obj.get_state())


class TestClassSave:

    def test_class_save(self):
        pass


class TestClassLoad:

    def test_class_load(self):
        pass


class TestModuleSave:

    def test_save_single_object(self, basic_object):
        pass

    def test_save_nested_object(self, nested_object):
        pass

    def test_save_pytorch_nested_alternating(self, alternating_nn_module_with_state):
        pass


class TestModuleLoad:

    def test_load_directory_single_file(self, basic_object):
        pass

    def test_load_nested_directory(self, nested_object):
        pass

    def test_load_pytorch_alternating(self, alternating_nn_module_with_state):
        pass


class TestSerializationIntegration:

    def test_state_and_load_roundtrip_single_object(self, basic_object):
        old_obj = basic_object(from_config=True)
        state = old_obj.get_state()
        new_obj = basic_object(from_config=False)
        new_obj.load_state(state, strict=False)
        assert old_obj.get_state() == new_obj.get_state()

    # def test_state_and_load_roundtrip_nested_object(self):
    #     pass

    def test_state_and_load_roundtrip_pytorch_alternating(self, alternating_nn_module_with_state):
        old_obj = alternating_nn_module_with_state(from_config=True)
        state = old_obj.get_state()
        new_obj = alternating_nn_module_with_state(from_config=False)
        new_obj.load_state(state, strict=False)
        old_state = old_obj.get_state()
        new_state = new_obj.get_state()
        check_mapping_equivalence(new_state, old_state)
        check_mapping_equivalence(old_state._metadata, new_state._metadata, exclude_config=True)

    # def test_class_save_and_load_roundtrip():
    #     pass
    #
    # def test_class_save_and_load_roundtrip_nested():
    #     pass
    #
    # def test_class_save_and_load_roundtrip_pytorch():
    #     pass

    def test_save_to_file_and_load_from_file_roundtrip(self, basic_object):
        old_obj = basic_object(from_config=True)
        state = old_obj.get_state()
        with tempfile.TemporaryDirectory() as path:
            save_state_to_file(state, path)
            state = load_state_from_file(path)
        new_obj = basic_object(from_config=False)
        new_obj.load_state(state, strict=False)
        old_state = old_obj.get_state()
        new_state = new_obj.get_state()
        check_mapping_equivalence(new_state, old_state)
        check_mapping_equivalence(old_state._metadata, new_state._metadata, exclude_config=True)

    def test_save_to_file_and_load_from_file_roundtrip_pytorch(self, alternating_nn_module_with_state):
        old_obj = alternating_nn_module_with_state(from_config=False)
        state = old_obj.get_state()
        with tempfile.TemporaryDirectory() as path:
            save_state_to_file(state, path)
            state = load_state_from_file(path)
        new_obj = alternating_nn_module_with_state(from_config=False)
        new_obj.load_state(state, strict=False)
        old_state = old_obj.get_state()
        new_state = new_obj.get_state()
        check_mapping_equivalence(new_state, old_state)
        check_mapping_equivalence(old_state._metadata, new_state._metadata, exclude_config=False)

    def test_save_to_file_and_load_from_file_roundtrip_complex(self, complex_multi_layered):
        TORCH_TAG_PREFIX = "torch"
        make_component(torch.nn.Module, TORCH_TAG_PREFIX, only_module='torch.nn')
        old_obj = complex_multi_layered(from_config=True)
        # Test that the current state is actually saved, for a
        # Component-only child of torch objects
        old_obj.child.child.child.x = 24
        state = old_obj.get_state()
        with tempfile.TemporaryDirectory() as path:
            save_state_to_file(state, path)
            list_files(path)
            state_loaded = load_state_from_file(path)
            check_mapping_equivalence(state, state_loaded)
            # assert False
        new_obj = complex_multi_layered(from_config=True)
        new_obj.load_state(state_loaded, strict=False)
        old_state = old_obj.get_state()
        new_state = new_obj.get_state()
        check_mapping_equivalence(new_state, old_state)
        check_mapping_equivalence(old_state._metadata, new_state._metadata, exclude_config=False)

    @pytest.mark.parametrize("pickle_only", [True, False])
    @pytest.mark.parametrize("compress_save_file", [True, False])
    def test_save_to_file_and_load_from_file_roundtrip_complex_nontorch_root(self,
            complex_multi_layered_nontorch_root, pickle_only, compress_save_file):
        TORCH_TAG_PREFIX = "torch"
        make_component(torch.nn.Module, TORCH_TAG_PREFIX, only_module='torch.nn')
        old_obj = complex_multi_layered_nontorch_root(from_config=True)
        state = old_obj.get_state()
        with tempfile.TemporaryDirectory() as root_path:
            path = os.path.join(root_path, 'savefile.flambe')
            save_state_to_file(state, path, compress_save_file, pickle_only)
            list_files(path)
            print("\n\nHello world\n\n")
            if pickle_only:
                path += '.pkl'
            if compress_save_file:
                path += '.tar.gz'
            state_loaded = load_state_from_file(path)
            print("original state: ", state)
            print("loaded state: ", state_loaded)
            check_mapping_equivalence(state, state_loaded)
            check_mapping_equivalence(state._metadata, state_loaded._metadata)
        new_obj = complex_multi_layered_nontorch_root(from_config=True)
        int_state = new_obj.get_state()
        new_obj.load_state(state_loaded, strict=False)
        old_state = old_obj.get_state()
        new_state = new_obj.get_state()
        check_mapping_equivalence(new_state, old_state)
        check_mapping_equivalence(old_state._metadata, new_state._metadata)
        check_mapping_equivalence(int_state._metadata, state_loaded._metadata)

    @pytest.mark.parametrize("pickle_only", [True, False])
    @pytest.mark.parametrize("compress_save_file", [True, False])
    def test_module_save_and_load_roundtrip(self, basic_object, pickle_only, compress_save_file):
        old_obj = basic_object(from_config=True)
        with tempfile.TemporaryDirectory() as root_path:
            path = os.path.join(root_path, 'savefile.flambe')
            save(old_obj, path, compress_save_file, pickle_only)
            if pickle_only:
                path += '.pkl'
            if compress_save_file:
                path += '.tar.gz'
            new_obj = load(path)
        old_state = old_obj.get_state()
        new_state = new_obj.get_state()
        check_mapping_equivalence(new_state, old_state)
        check_mapping_equivalence(old_state._metadata, new_state._metadata, exclude_config=False)

    @pytest.mark.parametrize("pickle_only", [True, False])
    @pytest.mark.parametrize("compress_save_file", [True, False])
    def test_module_save_and_load_roundtrip_pytorch(self,
                                                    alternating_nn_module_with_state,
                                                    pickle_only,
                                                    compress_save_file):
        old_obj = alternating_nn_module_with_state(from_config=True)
        with tempfile.TemporaryDirectory() as root_path:
            path = os.path.join(root_path, 'savefile.flambe')
            save(old_obj, path, compress_save_file, pickle_only)
            if pickle_only:
                path += '.pkl'
            if compress_save_file:
                path += '.tar.gz'
            new_obj = load(path)
        old_state = old_obj.get_state()
        new_state = new_obj.get_state()
        check_mapping_equivalence(new_state, old_state)
        check_mapping_equivalence(old_state._metadata, new_state._metadata, exclude_config=False)

    # def test_module_save_and_load_example_encoder(self):
    #     TORCH_TAG_PREFIX = "torch"
    #     make_component(torch.nn.Module, TORCH_TAG_PREFIX, only_module='torch.nn')
    #     make_component(torch.optim.Optimizer, TORCH_TAG_PREFIX, only_module='torch.optim')
    #     trainer = yaml.load(EXAMPLE_TRAINER_CONFIG)()
    #     with tempfile.TemporaryDirectory() as path:
    #         save(trainer, path)
    #         loaded_trainer = load(path)
    #     old_state = trainer.get_state()
    #     new_state = loaded_trainer.get_state()
    #     check_mapping_equivalence(new_state, old_state)
    #     check_mapping_equivalence(old_state._metadata, new_state._metadata, exclude_config=False)

    def test_module_save_and_load_single_instance_appears_twice(self, make_classes_2):
        txt = """
!C
one: !A
  akw2: &theb !B
    bkw2: test
    bkw1: 1
  akw1: 8
two: !A
  akw1: 8
  # Comment Here
  akw2: *theb
"""
        c = yaml.load(txt)()
        c.one.akw2.bkw1 = 6
        assert c.one.akw2 is c.two.akw2
        assert c.one.akw2.bkw1 == c.two.akw2.bkw1
        with tempfile.TemporaryDirectory() as path:
            save(c, path)
            state = load_state_from_file(path)
            loaded_c = load(path)
        assert loaded_c.one.akw2 is loaded_c.two.akw2
        assert loaded_c.one.akw2.bkw1 == loaded_c.two.akw2.bkw1


class TestSerializationExtensions:

    EXTENSIONS = {
        "ext1": "my_extension_1",
        "ext2": "my_extension_2",
        "ext3": "my_extension_3",
    }

    @pytest.mark.parametrize("pickle_only", [True, False])
    @pytest.mark.parametrize("compress_save_file", [True, False])
    @mock.patch('flambe.compile.serialization.is_installed_module')
    @mock.patch('flambe.compile.serialization.import_modules')
    @mock.patch('flambe.compile.component.Schema.add_extensions_metadata')
    def test_save_to_file_and_load_from_file_with_extensions(
            self, mock_add_extensions,
            mock_import_module, mock_installed_module,
            compress_save_file, pickle_only, schema):

        mock_installed_module.return_value = True

        schema_obj = schema()

        # Add extensions manually because if we use add_extensions_metadata
        # then no extensions will be added as the schema doesn't container_folder
        # any prefix.
        schema_obj._extensions = TestSerializationExtensions.EXTENSIONS

        obj = schema_obj()
        state = obj.get_state()

        with tempfile.TemporaryDirectory() as root_path:
            path = os.path.join(root_path, 'savefile.flambe')
            save_state_to_file(state, path, compress_save_file, pickle_only)

            os.system(f"cat {path}/config.yaml")

            list_files(path)
            if pickle_only:
                path += '.pkl'
            if compress_save_file:
                path += '.tar.gz'
            state_loaded = load_state_from_file(path)

            check_mapping_equivalence(state, state_loaded)
            check_mapping_equivalence(state._metadata, state_loaded._metadata)

            _ = Basic.load_from_path(path)
            mock_add_extensions.assert_called_once_with(TestSerializationExtensions.EXTENSIONS)

