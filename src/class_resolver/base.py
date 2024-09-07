# -*- coding: utf-8 -*-

"""A base resolver."""

import logging
import sys
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Set,
)

if sys.version_info[:2] >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

from .utils import Hint, OptionalKwargs, X, Y, make_callback, normalize_string

if TYPE_CHECKING:
    import click.decorators
    import optuna

__all__ = [
    "BaseResolver",
    "RegistrationError",
    "RegistrationNameConflict",
    "RegistrationSynonymConflict",
]

logger = logging.getLogger(__name__)


class RegistrationError(KeyError, Generic[X], ABC):
    """Raised when trying to add a new element to a resolver with a pre-existing lookup key."""

    def __init__(self, resolver: "BaseResolver[X, Y]", key: str, proposed: X, label: str):
        """Initialize the registration error.

        :param resolver: The resolver where the registration error occurred
        :param key: The key (either in the ``lookup_dict`` or ``synonyms``) where the conflict occurred
        :param proposed: The proposed overwrite on the given key
        :param label: The origin of the error (either "name" or "synonym")
        """
        self.resolver = resolver
        self.key = key
        self.proposed = proposed
        self.label = label
        self.existing = self._get_existing()

    @abstractmethod
    def _get_existing(self) -> X:
        """Get the pre-existing element based on the error type and the given key."""

    def __str__(self) -> str:
        """Coerce the registration error to a string."""
        return (
            f"Conflict on registration of {self.label} {self.key}:\n"
            f"Existing: {self.existing}\n"
            f"Proposed: {self.proposed}"
        )


class RegistrationNameConflict(RegistrationError[X]):
    """Raised on a conflict with the lookup dict."""

    def _get_existing(self) -> X:
        return self.resolver.lookup_dict[self.key]


class RegistrationSynonymConflict(RegistrationError[X]):
    """Raised on a conflict with the synonym dict."""

    def _get_existing(self) -> X:
        return self.resolver.synonyms[self.key]


class BaseResolver(ABC, Generic[X, Y]):
    """A resolver for arbitrary elements.

    This class is parametrized by two variables:

    - ``X`` is the type of element in the resolver
    - ``Y`` is the type that gets made by the ``make`` function. This is typically
      the same as ``X``, but might be different from ``X``, such as in the class resolver.
    """

    default: Optional[X]
    #: The mapping from synonyms to the classes indexed by this resolver
    synonyms: Dict[str, X]
    #: The mapping from normalized class names to the classes indexed by this resolver
    lookup_dict: Dict[str, X]
    #: The shared suffix fo all classes derived from the base class
    suffix: Optional[str]

    def __init__(
        self,
        elements: Optional[Iterable[X]] = None,
        *,
        default: Optional[X] = None,
        synonyms: Optional[Mapping[str, X]] = None,
        suffix: Optional[str] = None,
    ):
        """Initialize the resolver.

        :param elements: The elements to register
        :param default: The optional default element
        :param synonyms: The optional synonym dictionary
        :param suffix: The optional shared suffix of all instances
        """
        self.default = default
        self.synonyms = dict(synonyms or {})
        self.lookup_dict = {}
        self.suffix = suffix
        if elements is not None:
            for element in elements:
                self.register(element)

    def __iter__(self) -> Iterator[X]:
        """Iterate over the registered elements."""
        return iter(self.lookup_dict.values())

    @property
    def options(self) -> Set[str]:
        """Return the normalized option names."""
        return set(self.lookup_dict.keys()).union(self.synonyms.keys())

    @abstractmethod
    def extract_name(self, element: X) -> str:
        """Get the name for an element."""

    def extract_synonyms(self, element: X) -> Collection[str]:
        """Get synonyms from an element."""
        return []

    def normalize(self, s: str) -> str:
        """Normalize the string with this resolve's suffix."""
        return normalize_string(s, suffix=self.suffix)

    def register(
        self,
        element: X,
        synonyms: Optional[Iterable[str]] = None,
        raise_on_conflict: bool = True,
    ) -> None:
        """Register an additional element with this resolver.

        :param element: The element to register
        :param synonyms: An optional iterable of synonyms to add for the element
        :param raise_on_conflict: Determines the behavior when a conflict is encountered on either
            the normalized element name or a synonym. If true, will raise an exception. If false, will
            simply disregard the entry.

        :raises RegistrationNameConflict: If ``raise_on_conflict`` is true
            and there's a conflict with the lookup dict
        :raises RegistrationSynonymConflict: If ``raise_on_conflict`` is true
            and there's a conflict with the synonym dict
        :raises ValueError: If any given synonyms are empty strings
        """
        key = self.normalize(self.extract_name(element))
        if key not in self.lookup_dict and key not in self.synonyms:
            self.lookup_dict[key] = element
        elif key in self.lookup_dict and raise_on_conflict:
            raise RegistrationNameConflict(self, key, element, label="name")
        elif key in self.synonyms and raise_on_conflict:
            raise RegistrationSynonymConflict(self, key, element, label="name")

        _synonyms = set(synonyms or [])
        _synonyms.update(self.extract_synonyms(element))

        for synonym in _synonyms:
            synonym_key = self.normalize(synonym)
            if not synonym_key:
                raise ValueError(f"Tried to use empty synonym for {element}")
            if synonym_key not in self.synonyms and synonym_key not in self.lookup_dict:
                self.synonyms[synonym_key] = element
            elif synonym_key in self.lookup_dict and raise_on_conflict:
                raise RegistrationNameConflict(self, synonym_key, element, label="synonym")
            elif synonym_key in self.synonyms and raise_on_conflict:
                raise RegistrationSynonymConflict(self, synonym_key, element, label="synonym")

    @abstractmethod
    def lookup(self, query: Hint[X], default: Optional[X] = None) -> X:
        """Lookup an element."""

    def docdata(self, query: Hint[X], *path: str, default: Optional[X] = None) -> Any:
        """Lookup an element and get its docdata.

        :param query: The hint for looking something up in the resolver
            passed to :func:`lookup`
        :param path: An optional path for traversing the resulting docdata
            dictionary
        :param default: The default value to pass to :func:`lookup`
        :returns: The optional docdata retrieved with :func:`docdata.get_docdata`
        """
        from docdata import get_docdata

        x = self.lookup(query=query, default=default)
        rv = get_docdata(x)
        for part in path:
            rv = rv[part]
        return rv

    @abstractmethod
    def make(
        self,
        query: Hint[X],
        pos_kwargs: OptionalKwargs = None,
        **kwargs: Any,
    ) -> Y:
        """Make an element."""

    def make_safe(
        self, query: Hint[X], pos_kwargs: OptionalKwargs = None, **kwargs: Any
    ) -> Optional[Y]:
        """Run make, but pass through a none query."""
        if query is None:
            return None
        return self.make(query=query, pos_kwargs=pos_kwargs, **kwargs)

    def _default(self, default: Hint[X]) -> X:
        if default is not None:
            if isinstance(default, str):
                return self.lookup(default)
            else:
                return default
        elif self.default is not None:
            return self.default
        else:
            raise ValueError("no default given either from resolver or explicitly")

    def get_option(
        self,
        *flags: str,
        default: Hint[X] = None,
        as_string: bool = False,
        required: bool = False,
        **kwargs: Any,
    ) -> Callable[["click.decorators.FC"], "click.decorators.FC"]:
        """Get a click option for this resolver."""
        if not required:
            norm_default = self._default(default)
            looked_up = self.lookup(norm_default)
            name = self.extract_name(looked_up)
            key = self.normalize(name)
        else:
            key = None

        import click

        # TODO are there better ways to type options?
        return click.option(  # type:ignore
            *flags,
            type=click.Choice(list(self.lookup_dict), case_sensitive=False),
            default=[key] if kwargs.get("multiple") else key,
            show_default=True,
            callback=None if as_string else make_callback(self.lookup),
            required=required,
            **kwargs,
        )

    def register_entrypoint(self, group: str) -> None:
        """Register additional entries from an entrypoint."""
        for element in self._from_entrypoint(group).difference(self.lookup_dict.values()):
            self.register(element)

    @staticmethod
    def _from_entrypoint(group: str) -> Set[X]:
        elements: Set[X] = set()
        for entry in entry_points(group=group):
            try:
                element = entry.load()
            except (ImportError, AttributeError):
                logger.warning("could not load %s", entry.name)
            else:
                elements.add(element)
        return elements

    @classmethod
    def from_entrypoint(cls, group: str, **kwargs: Any) -> "BaseResolver[X, Y]":
        """Make a resolver from the elements registered at the given entrypoint."""
        elements = cls._from_entrypoint(group)
        return cls(elements, **kwargs)

    def optuna_lookup(self, trial: "optuna.Trial", name: str) -> X:
        """Suggest an element from this resolver for hyper-parameter optimization in Optuna.

        :param trial: A trial object from :mod:`optuna`. Note that this object shouldn't be constructed
            by the developer, and should only get constructed inside the optuna framework when
            using :meth:`optuna.Study.optimize`.
        :param name: The name of the `param` within an optuna study.
        :returns: An element chosen by optuna, then run through :func:`lookup`.

        In the following example, Optuna is used to determine the best classification
        algorithm from scikit-learn when applied to the famous iris dataset.

        .. code-block::

            import optuna
            from sklearn import datasets
            from sklearn.model_selection import train_test_split

            from class_resolver.contrib.sklearn import classifier_resolver


            def objective(trial: optuna.Trial) -> float:
                x, y = datasets.load_iris(return_X_y=True)
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.33, random_state=42,
                )
                clf_cls = classifier_resolver.optuna_lookup(trial, "model")
                clf = clf_cls()
                clf.fit(x_train, y_train)
                return clf.score(x_test, y_test)


            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=100)
        """
        key = trial.suggest_categorical(name, sorted(self.lookup_dict))
        return self.lookup(key)
