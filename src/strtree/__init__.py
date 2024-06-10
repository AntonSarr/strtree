__doc__ = """
strtree - a Python package for strings binary classification, based on trees and regular expressions
====================================================================================================

With strtree you can:
- Automatically find shortest regular expressions matchings strings from a positive class
- Perform a binary classification of strings with regular expressions and desired level of precision
"""

hard_dependencies = (
    "numpy",
)
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies


from .utils import Pattern
from .utils import PatternNode
from .utils import StringTree

