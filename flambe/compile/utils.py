from typing import Type, Set, Any, Optional, List

from urllib.parse import urlparse


def all_subclasses(class_: Type[Any]) -> Set[Type[Any]]:
    """Return a set of all subclasses for a given class object

    Recursively collects all subclasses of `class_` down the object
    hierarchy into one set.

    Parameters
    ----------
    class_ : Type[Any]
        Class to retrieve all subclasses for

    Returns
    -------
    Set[Type[Any]]
        All subclasses of class_

    """
    subsubclasses = set([s for c in class_.__subclasses__() for s in all_subclasses(c)])
    return set(class_.__subclasses__()).union(subsubclasses)


def make_component(class_: type,
                   tag_namespace: Optional[str] = None,
                   only_module: Optional[str] = None,
                   parent_component_class: Optional[Type] = None,
                   exclude: Optional[List[str]] = None) -> None:
    """Make class and all its children a `Component`

    For example a call to `make_component(torch.optim.Adam, "torch")`
    will make the tag `!torch.Adam` accessible in any yaml configs.
    This does *NOT* monkey patch (aka swizzle) torch, but instead
    creates a dynamic subclass which will be used by the yaml
    constructor i.e. only classes loaded from the config will be
    affected, anything imported and used in code.

    Parameters
    ----------
    class_ : type
        To be registered with yaml as a component, along with all its
        children
    tag_prefix : str
        Added to beginning of all the tags
    only_module : str
        Module prefix used to limit the scope of what gets registered
    parent_component_class : Type
        Parent class to use for creating a new component class; should
        be a subclass of :class:``~flambe.compile.Component`` (defaults
        to ``Component``)
    exclude: List[str], optional
        A list of modules to ignore

    Returns
    -------
    None

    """
    from flambe.compile import dynamic_component, Component
    if parent_component_class is None:
        parent_component_class = Component
    elif not issubclass(parent_component_class, Component):
        raise Exception("Only a subclass of Component should be used for 'parent_component_class'")
    for subclass in all_subclasses(class_):
        if exclude and any(ex in subclass.__module__ for ex in exclude):
            continue
        if only_module is None or subclass.__module__.startswith(only_module):
            dynamic_component(subclass, tag=subclass.__name__, tag_namespace=tag_namespace,
                              parent_component_class=parent_component_class)


def _is_url(resource: str) -> bool:
    """Whether a given resource is a remote URL.

    Resolve by searching for a scheme.

    Parameters
    ----------
    resource: str
        The given resource

    Returns
    -------
    bool
        If the resource is a remote URL.

    """
    scheme = urlparse(resource).scheme
    return scheme != ''
