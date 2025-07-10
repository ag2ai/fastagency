import functools
import importlib
import inspect
import pkgutil
import random
import sys
import types
from asyncio import iscoroutinefunction
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import pytest
import pytest_asyncio

__all__ = ["add_random_suffix", "parametrize_fixtures", "tag", "tag_list"]


def add_random_suffix(prefix: str) -> str:
    return f"{prefix}_{random.randint(0, 1_000_000_000):09d}"


F = TypeVar("F", bound=Callable[..., Any])

_tags: dict[str, list[str]] = {}


def tag(*args: str) -> Callable[[F], F]:
    def decorator(f: F, args: tuple[str, ...] = args) -> F:
        global _tags
        if not hasattr(f, "_pytestfixturefunction"):
            raise ValueError(f"function {f.__name__} is not a fixture")

        name = f._pytestfixturefunction.name
        if name is None:
            name = f.__name__

        for my_tag in args:
            if my_tag in _tags:
                _tags[my_tag].append(name)
            else:
                _tags[my_tag] = [name]

        return f

    return decorator


def tag_list(*args: str) -> Callable[[list[F]], list[F]]:
    def decorator(fs: list[F], args: tuple[str, ...] = args) -> list[F]:
        return [tag(*args)(f) for f in fs]

    return decorator


def get_by_tag(*args: str) -> list[str]:
    xs = [_tags.get(my_tag, []) for my_tag in args]
    return list(functools.reduce(set.intersection, map(set, xs)))  # type: ignore[arg-type]


def get_tags() -> list[str]:
    return list(_tags.keys())


def get_caller_globals() -> dict[str, Any]:
    # Get the caller's frame
    caller_frame = inspect.stack()[2].frame

    # Set the global variable in the caller's module
    caller_globals = caller_frame.f_globals

    return caller_globals


def parametrize_fixtures(
    parameter_name: str, src_fixtures: list[str]
) -> Callable[[F], F]:
    def decorator(f: F, parameter_name: str = parameter_name) -> F:
        f = pytest.mark.parametrize(parameter_name, src_fixtures, indirect=True)(f)

        # this is needed to make the fixture available in the caller's module
        @pytest.fixture(name=parameter_name)
        def wrapper(request: Any) -> Any:
            return request.getfixturevalue(request.param)

        caller_globals = get_caller_globals()

        var_name = add_random_suffix(f"parametrized_fixtures_{parameter_name}")
        caller_globals[var_name] = wrapper

        return f

    return decorator


def rename_parameter(src_name: str, dst_name: str) -> Callable[[F], F]:
    def decorator(f: F) -> F:
        # Get the original signature of the function
        sig = inspect.signature(f)

        # Create a new parameter list with src_name replaced by dst_name
        params = [
            inspect.Parameter(
                dst_name if param.name == src_name else param.name,
                param.kind,
                default=param.default,
                annotation=param.annotation,
            )
            for param in sig.parameters.values()
        ]

        # Create a new signature with the modified parameters
        new_sig = sig.replace(parameters=params)

        # Define the body of the new function
        if iscoroutinefunction(f):

            async def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
                bound_args = new_sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                arguments = bound_args.arguments

                if dst_name in arguments:
                    arguments[src_name] = arguments.pop(dst_name)

                return await f(**arguments)
        else:

            def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
                bound_args = new_sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                arguments = bound_args.arguments

                if dst_name in arguments:
                    arguments[src_name] = arguments.pop(dst_name)

                return f(**arguments)

        # Create the new function with the modified signature
        new_func = types.FunctionType(
            wrapper.__code__,
            globals(),
            name=f.__name__,
            argdefs=wrapper.__defaults__,
            closure=wrapper.__closure__,
        )
        new_func.__signature__ = new_sig  # type: ignore[attr-defined]
        functools.update_wrapper(new_func, f)
        return new_func  # type: ignore

    return decorator


def expand_fixture(
    dst_fixture_prefix: str,
    src_fixtures_names: list[str],
    placeholder_name: str,
) -> Callable[[F], list[F]]:
    def decorator(f: F) -> list[F]:
        retval: list[F] = []
        for src_type in src_fixtures_names:
            name = f"{dst_fixture_prefix}_{src_type}"

            f_renamed = rename_parameter(placeholder_name, src_type)(f)
            if iscoroutinefunction(f):
                f_fixture = pytest_asyncio.fixture(name=name)(f_renamed)
            else:
                f_fixture = pytest.fixture(name=name)(f_renamed)

            caller_globals = get_caller_globals()
            caller_globals[name] = f_fixture

            retval.append(f_fixture)  # type: ignore[arg-type]

        return retval

    return decorator


@contextmanager
def add_to_sys_path(path: Optional[Path]) -> Iterator[None]:
    if path is None:
        yield
        return

    if not path.exists():
        raise ValueError(f"Path {path} does not exist")

    sys.path.append(str(path))
    try:
        yield
    finally:
        sys.path.remove(str(path))


def list_submodules(
    module_name: str, *, include_path: Optional[Path] = None, include_root: bool = True
) -> list[str]:
    """List all submodules of a given module.

    Args:
        module_name (str): The name of the module to list submodules for.
        include_path (Optional[Path], optional): The path to the module. Defaults to None.
        include_root (bool, optional): Whether to include the root module in the list. Defaults to True.

    Returns:
        list: A list of submodule names.
    """
    with add_to_sys_path(include_path):
        try:
            module = importlib.import_module(module_name)  # nosemgrep
        except Exception:
            return []

        # Get the path of the module. This is necessary to find its submodules.
        module_path = module.__path__

        # Initialize an empty list to store the names of submodules
        submodules = [module_name] if include_root else []

        # Iterate over the submodules in the module's path
        for _, name, ispkg in pkgutil.iter_modules(
            module_path, prefix=f"{module_name}."
        ):
            # Add the name of each submodule to the list
            submodules.append(name)

            if ispkg:
                submodules.extend(list_submodules(name, include_root=False))

        # Return the list of submodule names
        return submodules
