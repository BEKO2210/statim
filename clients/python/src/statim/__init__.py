"""Official Python client for the Statim decision API.

``Client`` talks to ``POST /v1/systemone``. Question type ``noul`` comes back
as :class:`YesNoAnswer`. The package has no runtime dependencies beyond the
standard library.
"""

from statim._version import __version__
from statim.client import Client
from statim.errors import (
    AuthenticationError,
    BadRequestError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    StatimError,
    TransportError,
    UnprocessableEntityError,
)
from statim.types import (
    Action,
    BatchResult,
    ChoiceAnswer,
    Decision,
    Health,
    Model,
    ModelList,
    Ready,
    Routing,
    ScoreAnswer,
    Usage,
    YesNoAnswer,
)

__all__ = [
    "Action",
    "AuthenticationError",
    "BadRequestError",
    "BatchResult",
    "ChoiceAnswer",
    "Client",
    "Decision",
    "Health",
    "Model",
    "ModelList",
    "PayloadTooLargeError",
    "Ready",
    "Routing",
    "ScoreAnswer",
    "ServiceUnavailableError",
    "StatimError",
    "TransportError",
    "UnprocessableEntityError",
    "Usage",
    "YesNoAnswer",
    "__version__",
]
