"""
This folder implements a small python wrapper around metatab APIà
Metatab was mainly designed to work via CLI. This means that it has not a clean API to expose.
This wrapper address this.

Disclaimer: This solution is not optimal since it introduces a lot of logic convolution and 
deplucation. A greatly better solution is to refactor the estimators classes to python classes.
Even though this is not too complex by itself, then one should adapt the whole CLI API, the
family ensembler and the tests.

This solution is a bad fix. In future the best thing could be dropping the whole CLI API, 
the family ensmebler and change the tests to have a smooter transition. 
The new API should really follow this tiny wrapper logic, and the design of an initial attempt of
refactoring available at the "refactor" branch. The key is to drop the user APIA constrains 
and secondary utilities while maintaining the core classes managining, preprocessing, optimization 
and ensembling 
"""