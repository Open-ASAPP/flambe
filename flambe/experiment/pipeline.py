import copy
from typing import Optional, Dict, List, Set, Tuple, Any

from flambe.compile import Schema, Link, Options
from flambe.search.search import Checkpoint


class Pipeline:

    def __init__(self,
                 schemas: Dict[str, Schema],
                 variant_ids: Optional[Dict[str, str]] = None,
                 checkpoints: Optional[Dict[str, Checkpoint]] = None):
        """Initialize a Pipeline.

        Parameters
        ----------
        schemas : Dict[str, Schema]
            [description]
        variant_ids : Optional[Dict[str, str]], optional
            [description], by default None
        checkpoints : Optional[Dict[str, Checkpoint]], optional
            [description], by default None

        """
        self.schemas = schemas
        self.var_ids = variant_ids if variant_ids is not None else dict()
        self.checkpoints = checkpoints if checkpoints is not None else dict()
        self.error = False
        self.metric = None

    @property
    def task(self) -> str:
        """Get the stage that will be executed.

        Returns
        -------
        str
            The stage name to be executed.

        """
        return list(self.schemas.keys())[-1]

    @property
    def dependencies(self) -> List[str]:
        """Get the dependencies for this pipeline.

        Returns
        -------
        List[str]
            A list of dependencies, as stage names.

        """
        return list(self.schemas.keys())[:-1]

    def set_param(self, path: Optional[Tuple[str]], value: Any) -> None:
        schema_name, path = path[0], path[1:]
        self.schemas[schema_name].set_param(path, value)

    def get_param(self, path: Optional[Tuple[str]]) -> None:
        schema_name, path = path[0], path[1:]
        return self.schemas[schema_name].get_param(path)

    def extract_search_space(self) -> Dict[Tuple[str, ...], Options]:
        search_space = {}
        for name, schema in self.schemas.items():
            for path, val in schema.extract_search_space():
                new_path = (name,) + path
                search_space[new_path] = val
        return search_space

    def extract_links(self) -> List[Link]:
        links = []
        for name, schema in self.schemas.items():
            for path, item in Schema.traverse(schema, yield_schema='never'):
                if isinstance(item, Link):
                    links.append(item)
        return links

    def set_from_search_space(self, search_space: Dict[Tuple[str, ...], Any]) -> None:
        for path, value in search_space.items():
            name, new_path = path[0], path[1:]
            schema = self.schemas[name]
            schema.set_param(new_path, value)

    def initialize(self):
        """Override intialization."""
        cache = {}
        for name, schema in self.schemas.items():
            obj = schema.initialize(path=(name,), cache=cache, root=self)
            if name in self.checkpoints:
                # TODO: what if we don't save states but the
                # full object? How do we handle links then?
                obj.load_state(self.checkpoints[name].get())

        return obj

    def sub_pipeline(self, stage_name: str) -> 'Pipeline':
        """Return subset of the pipeline stages ending in stage_name

        The subset of pipeline stages will include all dependencies
        needed for stage the given stage and all their dependencies
        and so on.

        """

        def get_deps(schema: Schema, visited: Optional[Set[str]] = None) -> Set[str]:
            """[summary]

            Parameters
            ----------
            schema : Schema
                [description]
            visited : Optional[Set[str]], optional
                [description], by default None

            Returns
            -------
            Set[str]
                [description]

            """
            links = set()
            visited = visited if visited is not None else set()
            for path, item in Schema.traverse(schema, yield_schema='never'):
                if isinstance(item, Link):
                    stage = item.schematic_path[0]
                    links.add(stage)
                    if stage not in visited:
                        visited.add(stage)
                        new_links = get_deps(self.schemas[stage], visited)
                        links.update(new_links)
            return links

        deps = get_deps(self)
        sub_stages = {k: v for k, v in self.schemas.items() if k in deps}
        var_ids = {k: v for k, v in self.var_ids.items() if k in deps}
        checkpoints = {k: v for k, v in self.checkpoints.items() if k in deps}
        sub_stages[self.task] = self.schemas[self.task]
        return Pipeline(sub_stages, var_ids, checkpoints)

    def append(self,
               name: str,
               schema: Schema,
               var_id: Optional[str] = None,
               checkpoint: Optional[Checkpoint] = None):
        """Append a new schema.

        Parameters
        ----------
        schema : Schema
            [description]
        var_id : Optional[str], optional
            [description], by default None
        checkpoint : Optional[Checkpoint], optional
            [description], by default None

        """
        if name is self.schemas:
            raise ValueError("{name} already in pipeline.")

        self.schemas[name] = schema
        if var_id is not None:
            self.var_ids[name] = var_id
        if checkpoint is not None:
            self.checkpoints[name] = checkpoint

    def merge(self, other: 'Pipeline', in_place: bool = False) -> 'Pipeline':
        """Updates internals with the provided pipeline.

        Parameters
        ----------
        other: Pipeline
            The pipeline to merge in.

        in_place: bool, optional
            Whether to perform the merge in place. Default ``False``.

        Returns
        -------
        Pipeline
            A new merged pipeline.

        """
        pipeline = self if in_place else copy.deepcopy(self)
        pipeline.schemas.update(other.schemas)
        pipeline.var_ids.update(other.var_ids)
        pipeline.checkpoints.update(other.checkpoints)
        return pipeline

    def flatten(self) -> 'Pipeline':
        """Check for matches.

        Returns
        -------
        Pipeline
            A flattened versiion of this pipeline.

        """
        schemas, checkpoints, var_ids = {}, {}, {}
        for name, schema in self.schemas.items():
            if isinstance(schema, Pipeline):
                schemas.update(schema.schemas)
                checkpoints.update(schema.checkpoints)
                var_ids.update(schema.var_ids)
            else:
                schemas[name] = schema

        return Pipeline(schemas, var_ids, checkpoints)

    def matches(self, other: 'Pipeline') -> bool:
        """Check for matches.

        Parameters
        ----------
        other : [type]
            [description]

        Returns
        -------
        bool
            [description]

        """
        for key in self.schemas.keys():
            if key in other.schemas and self.var_ids[key] != other.var_ids[key]:
                return False
        return True
