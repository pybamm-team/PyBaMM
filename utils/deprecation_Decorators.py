import functools
import warnings


def deprecate_arguments(deprecated_args, deprecated_in, removed_in, current_version):
    """
    Custom decorator to deprecate specific function arguments.

    Args:
        deprecated_args (dict): Dictionary of deprecated argument names with details.
            Example: {"old_arg": "Use 'new_arg' instead."}
        deprecated_in (str): Version when the argument was deprecated.
        removed_in (str): Version when the argument will be removed.
        current_version (str): Current version of the package/module.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for arg, message in deprecated_args.items():
                if arg in kwargs:
                    warnings.warn(
                        f"Argument '{arg}' is deprecated since version {deprecated_in} "
                        f"and will be removed in version {removed_in}. (Current version: {current_version}) "
                        f"{message}",
                        category=DeprecationWarning,
                        stacklevel=2,
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator
