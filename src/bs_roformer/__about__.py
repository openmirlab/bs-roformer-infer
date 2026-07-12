"""Single source of truth for the package version.

pyproject.toml reads it via hatch's dynamic version ([tool.hatch.version]);
__init__.py re-exports it. Bump the version here and nowhere else.

Reads: nothing (leaf module by design)
"""

__version__ = "0.1.5"
