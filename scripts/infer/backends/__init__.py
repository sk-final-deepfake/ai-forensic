from .base import OpticalFlowBackend
from .gmflow_backend import GmflowBackend
from .raft_backend import RaftBackend

BACKENDS = {
    "raft": RaftBackend,
    "gmflow": GmflowBackend,
}

__all__ = [
    "OpticalFlowBackend",
    "RaftBackend",
    "GmflowBackend",
    "BACKENDS",
]
