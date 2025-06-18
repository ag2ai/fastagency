import sys
from typing import Any

from ...base import CreateWorkflowUIMixin
from ...exceptions import FastAgencyCLIPythonVersionError
from ...helpers import check_imports
from ...messages import MessageProcessorMixin

# Only check version when Mesop is actually being imported/used
# This allows the package to be installed on Python 3.9 but prevents usage of Mesop UI
if sys.version_info < (3, 10):
    # Instead of raising immediately, define a placeholder that raises on usage
    class MesopUI(MessageProcessorMixin, CreateWorkflowUIMixin):
        """Placeholder MesopUI that raises an error on Python 3.9."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize MesopUI - raises error on Python 3.9."""
            raise FastAgencyCLIPythonVersionError(
                "Mesop requires Python 3.10 or higher"
            )

    __all__ = ["MesopUI"]
else:
    # Python 3.10+: normal Mesop import
    check_imports(["mesop"], "mesop")
    from .mesop import MesopUI

    __all__ = ["MesopUI"]
