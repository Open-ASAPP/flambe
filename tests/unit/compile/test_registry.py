import pytest

from flambe.compile.registry import get_registry, ROOT_NAMESPACE


@pytest.fixture
def registry():
    registry = get_registry()
    registry.reset()
    return registry


class TestRegistryCreate:

    def test_create_entry_for_class(self, registry):

        class A:
            pass

        registry.create(A, from_yaml=lambda x, y: None, to_yaml=lambda x, y: None)
        assert A in registry
        assert registry.default_tag(A) == 'A'

    def test_create_entry_for_class_with_tag(self, registry):

        class A:
            pass

        registry.create(A, tags=['a'], from_yaml=lambda x, y: None, to_yaml=lambda x, y: None)
        assert A in registry
        assert registry.default_tag(A) == 'a'

    def test_create_entry_for_class_with_factories(self, registry):

        class A:
            @classmethod
            def build(cls):
                return A()

        registry.create(A, factories=('build',), from_yaml=lambda x, y: None,
                        to_yaml=lambda x, y: None)
        assert A in registry
        entry = registry.namespaces[ROOT_NAMESPACE][A]
        assert len(entry.factories) == 1
        assert 'build' in entry.factories

    def test_create_entry_for_class_with_yaml_fns(self, registry):

        class A:
            pass

        def from_yaml_fn(constructor, node):
            pass

        def to_yaml_fn(representer, node):
            pass

        registry.create(A, from_yaml=from_yaml_fn, to_yaml=to_yaml_fn)
        assert A in registry

    def test_fail_entry_for_class_without_yaml_fns(self, registry):

        class A:
            pass

        with pytest.raises(ValueError):
            registry.create(A)

    def test_fail_entry_for_class_with_yaml_fns_wrong_signatures(self, registry):

        class A:
            pass

        def from_yaml_fn(constructor, node, something_else):
            pass

        def to_yaml_fn(representer):
            pass

        with pytest.raises(ValueError):
            registry.create(A, from_yaml=from_yaml_fn, to_yaml=to_yaml_fn)

    def test_create_entry_for_class_with_namespace(self, registry):

        class A:
            pass

        registry.create(A, namespace='tests', from_yaml=lambda x, y: None,
                        to_yaml=lambda x, y: None)
        assert A in registry

    def test_fail_duplicate_entry_for_class(self, registry):

        class A:
            pass

        registry.create(A, from_yaml=lambda x, y: None, to_yaml=lambda x, y: None)
        with pytest.raises(ValueError):
            registry.create(A, from_yaml=lambda x, y: None, to_yaml=lambda x, y: None)

    def test_fail_duplicate_entry_for_tag(self, registry):

        class A:
            pass

        class B:
            pass

        registry.create(A, tags=['x'], from_yaml=lambda x, y: None, to_yaml=lambda x, y: None)
        with pytest.raises(ValueError):
            registry.create(B, tags=['x'], from_yaml=lambda x, y: None, to_yaml=lambda x, y: None)

    def test_create_duplicate_entry_for_tag_different_namespaces(self, registry):

        class A:
            pass

        class B:
            pass

        registry.create(A, namespace='test1', tags=['x'], from_yaml=lambda x, y: None,
                        to_yaml=lambda x, y: None)
        registry.create(B, namespace='test2', tags=['x'], from_yaml=lambda x, y: None,
                        to_yaml=lambda x, y: None)
        assert A in registry
        assert B in registry

    def test_fail_duplicate_entry_for_class_different_namespaces(self, registry):

        class A:
            pass

        registry.create(A, namespace='test1', tags=['x'], from_yaml=lambda x, y: None,
                        to_yaml=lambda x, y: None)
        with pytest.raises(ValueError):
            registry.create(A, namespace='test2', from_yaml=lambda x, y: None,
                            to_yaml=lambda x, y: None)
        with pytest.raises(ValueError):
            registry.create(A, namespace='test2', tags=['y'], from_yaml=lambda x, y: None,
                            to_yaml=lambda x, y: None)
