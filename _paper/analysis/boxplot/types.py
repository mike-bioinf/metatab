from typing import TypeAlias, Any


# binary tuple with 2 values
simple_pair: TypeAlias = tuple[Any, Any]

# binary tuple of 2 other binary tuples
complex_pair: TypeAlias = tuple[tuple[Any, Any], tuple[Any, Any]]

# list of complex_pair elements
complex_pairs: TypeAlias = list[complex_pair]

# list of simple_pair elements
simple_pairs: TypeAlias = list[simple_pair]