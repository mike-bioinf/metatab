"""
This folder implements a small python wrapper around metatab internal API
Metatab was originally designed to work via CLI. This means that it has not a clean API to expose.
This wrapper address this by creating the interfaces for the exposed classes.
So the core idea here is to "artifically" built the estimator classes using mixins that creates the 
user API and a core class that does all the real job intercepting the cli original API.

Disclaimer: This solution is really BAD since it introduces a lot of logic convolution, classes duplication and a fragile python API. 
A greatly better solution is to refactor the "cli" estimators classes to python classes.
Even though this is not too complex by itself, then one should adapt the whole CLI API, the family ensembler and the tests.

In future the best thing could be dropping the whole CLI and family ensembler API.
The new API should/could follow the design of an initial attempt of refactoring available at the "refactor" branch. 
The key to do this is to maintain only the core classes/logic managining preprocessing, optimization and ensembling.
The rest can be cut.
"""