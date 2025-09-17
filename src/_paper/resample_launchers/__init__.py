"""
!!! NOT WORKING: 
Currently we are unable to pass the dict-like arguments 
correctly though the different steps. The shell calls peel off the quotes
and this causes a incorrect dict-like arg to be passed in the last call.
Generally the dict-like args are difficult to manage in the shell.
The best option is to separate them in different parameters/options.


Intended purpose:
Code to automatically launch the metatab-resample program in slurm jobs 
with the correct directivies (partition, n_cpus, n_gpus, ecc...)
depending on the inputs passed to the program (mainly based on the estimator).

We should move this code outside src since it's only for my personal use.
The problem is then to resolve the import issues when we use it. 
Infact if this subpackge is not under src then it is not installed 
and this will breaks the imports. 
For now it is git-ignored and no other file depends on it.
"""
