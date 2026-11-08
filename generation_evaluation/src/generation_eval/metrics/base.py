from dataclasses import dataclass, field
from importlib import metadata


@dataclass
class MetricOutcome:
    """What one metric returns: a 0-1 score and the working behind it."""

    score: float
    explanation: str
    details: dict = field(default_factory=dict)
    framework_version: str | None = None


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
