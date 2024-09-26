"""Utilities for automatic documentation."""

from __future__ import annotations

import importlib
import inspect
import textwrap
from collections import defaultdict
from typing import Callable, TypeVar

from .base import BaseResolver

__all__ = [
    "document_resolver",
    "DocKey",
]

F = TypeVar("F", bound=Callable)


class DocKey:
    """An object storing information about how a resolver is used in a signature."""

    name: str
    key: str
    resolver_path: str
    resolver: BaseResolver

    def __init__(
        self,
        name: str,
        resolver: str | BaseResolver,
        *,
        key: str | None = None,
    ) -> None:
        """Initialize the key for :func:`document_resolver`."""
        self.name = name
        self.key = f"{self.name}_kwargs" if key is None else key

        if isinstance(resolver, str):
            self.resolver_path = resolver
            module_name, variable_name = resolver.rsplit(".", 1)
            module = importlib.import_module(module_name)
            self.resolver = getattr(module, variable_name)
        elif isinstance(resolver, BaseResolver):
            raise NotImplementedError
        else:
            raise TypeError


def document_resolver(
    *keys: DocKey,
) -> Callable[[F], F]:
    """
    Build a decorator to add information about resolved parameter pairs.

    The decorator is intended for methods with follow the ``param`` + ``param_kwargs`` pattern and internally use a
    class resolver.

    .. code-block:: python

        from typing import Any, Union
        from torch import Tensor, nn
        from class_resolver import document_resolver, DocKey
        from class_resolver.contrib.torch import activation_resolver

        @document_resolver(
            DocKey("activation", "class_resolver.contrib.torch.activation_resolver")
        )
        def f(
            tensor: Tensor,
            activation: Union[None, str, type[nn.Module], nn.Module],
            activation_kwargs: dict[str, Any] | None,
        ):
            _activation = activation_resolver.make(activation, activation_kwargs)
            return _activation(tensor)


    This also can be stacked for multiple resolvers.

    .. code-block:: python

        from typing import Any, Union
        from torch import Tensor, nn
        from class_resolver import document_resolver
        from class_resolver.contrib.torch import activation_resolver, aggregation_resolver

        @document_resolver(
            DocKey("activation", "class_resolver.contrib.torch.activation_resolver"),
            DocKey("aggregation", "class_resolver.contrib.torch.aggregation_resolver"),
        )
        def f(
            tensor: Tensor,
            activation: Union[None, str, type[nn.Module], nn.Module],
            activation_kwargs: dict[str, Any] | None,
            aggregation: Union[None, str, type[nn.Module], nn.Module],
            aggregation_kwargs: dict[str, Any] | None,
        ):
            _activation = activation_resolver.make(activation, activation_kwargs)
            _aggregation = aggregation_resolver.make(aggregation, aggregation_kwargs)
            return _aggregation(_activation(tensor))

    If you have the resolvers handy, you can directly pass them.

    .. code-block:: python

        from typing import Any, Union
        from torch import Tensor, nn
        from class_resolver import document_resolver
        from class_resolver.contrib.torch import activation_resolver, aggregation_resolver

        @document_resolver(
            DocKey("activation", activation_resolver),
            DocKey("aggregation", aggregation_resolver),
        )
        def f(
            tensor: Tensor,
            activation: Union[None, str, type[nn.Module], nn.Module],
            activation_kwargs: dict[str, Any] | None,
            aggregation: Union[None, str, type[nn.Module], nn.Module],
            aggregation_kwargs: dict[str, Any] | None,
        ):
            _activation = activation_resolver.make(activation, activation_kwargs)
            _aggregation = aggregation_resolver.make(aggregation, aggregation_kwargs)
            return _aggregation(_activation(tensor))

    It might be the case that you have two different arguments that use the same resolver.
    No prob!

    .. code-block:: python

        from typing import Any, Union
        from torch import Tensor, nn
        from class_resolver import document_resolver
        from class_resolver.contrib.torch import activation_resolver, aggregation_resolver

        @document_resolver(
            DocKey("activation_1", activation_resolver),
            DocKey("activation_2", activation_resolver),
            DocKey("aggregation", aggregation_resolver),
        )
        def f(
            tensor: Tensor,
            activation_1: Union[None, str, type[nn.Module], nn.Module],
            activation_1_kwargs: dict[str, Any] | None,
            aggregation: Union[None, str, type[nn.Module], nn.Module],
            aggregation_kwargs: dict[str, Any] | None,
            activation_2: Union[None, str, type[nn.Module], nn.Module],
            activation_2_kwargs: dict[str, Any] | None,
        ):
            _activation_1 = activation_resolver.make(activation_1, activation_kwargs)
            _activation_2 = activation_resolver.make(activation_2, activation_kwargs)
            _aggregation = aggregation_resolver.make(aggregation, aggregation_kwargs)
            return _activation_2(_aggregation(_activation_2(tensor)))

    :param keys:
        A variadic list of keys, each describing:

        1. the names of the parameter
        2. the resolver used to construct a reference via the ``:data:`` role.
        3. the name of the parameter for giving keyword arguments. By default,
           this is constructed by taking the name and post-pending ``_kwargs``.

    :return:
        a decorator which extends a function's docstring.
    :raises ValueError:
        When either no parameter name was provided, there was a duplicate parameter name.
    """
    # input validation
    if not keys:
        raise ValueError("Must provided at least one parameter name.")

    # check for duplicates
    expanded_params = set(e for key in keys for e in (key.name, key.key))
    if len(expanded_params) < 2 * len(keys):
        raise ValueError(f"There are duplicates in (the expanded) {keys=}")

    # TODO: we could do some more sanitization, e.g., trying to match types, ...

    def add_note(func: F) -> F:
        """
        Extend the function's docstring with a note about resolved parameters.

        :param func:
            the function to decorate.

        :return:
            the function with extended docstring.

        :raises ValueError:
            When the signature does not contain the resolved parameter names, or the docstring is missing.
        """
        signature = inspect.signature(func)
        if missing := expanded_params.difference(signature.parameters):
            raise ValueError(f"{missing=} parameters in {signature=}.")
        if not func.__doc__:
            raise ValueError("docstring is empty")

        resolver_to_keys = defaultdict(list)
        for key in keys:
            if key.name not in signature.parameters:
                raise ValueError(
                    f"Gave document key for parameter {key.name}, which doesn't appear "
                    f"in the signature of {func.__name__}"
                )
            if key.key not in signature.parameters:
                raise ValueError(
                    f"Gave document key for parameter kwargs {key.name}, which doesn't "
                    f"appear in the signature of {func.__name__}"
                )
            resolver_to_keys[key.resolver_path].append(key)

        list_items = []
        for resolver_qualpath, subkeys in resolver_to_keys.items():
            pairs_str = ", ".join(f"``({key.name}, {key.key})``" for key in subkeys)
            list_item = f"- The parameter pairs {pairs_str} are used for :data:`{resolver_qualpath}`"
            list_items.append(list_item)

        note_str = textwrap.dedent(
            f"""\
            .. note ::

                {len(keys)} resolvers are used in this function.

                {"\n".join(list_items)}

                An explanation of resolvers and how to use them is given in
                https://class-resolver.readthedocs.io/en/latest/.
            """
        )
        note_str = textwrap.indent(text=note_str, prefix="        ", predicate=bool)
        # TODO: this is in-place
        func.__doc__ = f"{func.__doc__.lstrip()}\n\n{note_str}"
        return func

    return add_note
