"""Component registry for mapping strategies."""

from typing import Dict, Optional, Union

from mapping.protocols import Projector, Clusterer, AxisScorer

Component = Union[Projector, Clusterer, AxisScorer]


class ComponentRegistry:
    """Register and look up mapping components by name."""

    def __init__(self):
        self._projectors: Dict[str, Projector] = {}
        self._clusterers: Dict[str, Clusterer] = {}
        self._axis_scorers: Dict[str, AxisScorer] = {}

    def register(self, component: Component) -> None:
        """Register a component (replaces existing with same name)."""
        if isinstance(component, Projector):
            self._projectors[component.name] = component
        elif isinstance(component, Clusterer):
            self._clusterers[component.name] = component
        elif isinstance(component, AxisScorer):
            self._axis_scorers[component.name] = component
        else:
            raise TypeError(f"Unknown component type: {type(component)}")

    def get_projector(self, name: str) -> Optional[Projector]:
        return self._projectors.get(name)

    def get_clusterer(self, name: str) -> Optional[Clusterer]:
        return self._clusterers.get(name)

    def get_axis_scorer(self, name: str) -> Optional[AxisScorer]:
        return self._axis_scorers.get(name)

    @property
    def projector_names(self):
        return list(self._projectors.keys())

    @property
    def clusterer_names(self):
        return list(self._clusterers.keys())

    @property
    def axis_scorer_names(self):
        return list(self._axis_scorers.keys())
