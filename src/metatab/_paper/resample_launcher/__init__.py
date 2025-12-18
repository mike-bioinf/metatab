"""
Intended purpose:
Code to automatically launch the metatab-resample programs in slurm jobs 
with the correct directivies (partition, n_cpus, n_gpus, ecc...)
depending on the inputs passed to the program (mainly based on the estimator).

Note:
We should move this code outside src since it's only for my personal use.
The problem is then to resolve the import issues when we use it. 
Infact if this subpackge is not under src then it is not installed 
and this will breaks the imports.
"""
