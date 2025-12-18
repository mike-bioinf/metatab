import re
import sys
import subprocess
from metatab.cli.programs.resample.parser import parse_args
from metatab.ensemble.configuration import CollectionUserEnsembleConfiguration

from .constants import (
    CPU_ESTIMATORS, 
    ALL_GPU_ESTIMATORS,
    MULTIPLE_GPU_ESTIMATORS
)




class ResampleLauncher:
    '''
    Class to launch the "metatab-resample" programs.

    It is responsible for:
    - Constructing the Slurm resource instructions based on the input arguments.
    - Building the final "sbatch" command (including environment setup via the "setup_instruction").
    - Submitting the job.
    '''
    def __init__(self):
        self.args=sys.argv[1:]
        self.parsed_args=vars(parse_args(self.args))
        self.max_nthreads_on_parallel_partition = 10
        self.max_nthreads_on_gpus_partition = 10
        self.slurm_instructions: list[str] = None
        self.setup_instructions: str = (
            "ulimit -u 100000;"
            " source /lustre/home/epasolli/tools/miniconda3/bin/activate finetunetabpfn;"
        )


    def run(self):
        '''
        Create a new process in which the slurm job is launched.
        We spawn the new process via shell (shell=True in the run call)
        since otherwise we get file not found errors for the input program files.
        '''
        self._infer_slurm_instrunctions()

        script_body = (
            self.setup_instructions + 
            " metatab-resample " + 
            " ".join(self.args)
        )

        sbatch_cmd = (
            "sbatch " +
            " ".join(self.slurm_instructions) +
            f' --wrap="{script_body}"'
        )
        
        try:
            _ = subprocess.run(sbatch_cmd, check=True, shell=True)
        except Exception as e:
            raise ValueError(f"The launcher failed to launch the job with error: '{e}'")



    def _infer_slurm_instrunctions(self) -> None:
        if self.parsed_args["estimator_mode"] == "family_ensemble":
            configuration = self.parsed_args["configuration"]
            
            if re.match(r'^(all|cpu|gpu)_(meta|random)_\d+$', configuration):
                device, _, _ = configuration.split("_")
                device = "gpu" if device == "all" else device
            else:
                # we evaluate whether gpu-based estimators are requested
                conf_collection = CollectionUserEnsembleConfiguration.load_json(configuration)
                device = "cpu"
                for conf in conf_collection.configurations:
                    if conf.estimator in ALL_GPU_ESTIMATORS:
                        device = "gpu"
                        break
                
        else:
            estimator = self.parsed_args["estimator"]
            device = "cpu" if estimator in CPU_ESTIMATORS else "gpu"

        nthreads = self.parsed_args["nthreads"]

        if device == "cpu":
            if nthreads > self.max_nthreads_on_parallel_partition:
                raise ValueError((
                    "The number of user requested threads must be less" 
                    f"or equal than {self.max_nthreads_on_parallel_partition}."
                ))
            
            device_instructions = [
                "--partition=parallel",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={self.max_nthreads_on_parallel_partition}"
            ]

        else:
            if nthreads > self.max_nthreads_on_gpus_partition:
                raise ValueError((
                    "The number of user requested threads must be less"
                    f"or equal than {self.max_nthreads_on_gpus_partition}."
                ))
            
            n_gpus = 4 if estimator in MULTIPLE_GPU_ESTIMATORS else 1

            device_instructions = [
                "--partition=gpus",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={self.max_nthreads_on_gpus_partition}",
                f"--gpus-per-node={n_gpus}", # equivalent f"--gres=gpu:{n_gpus}""
                "--mem-per-gpu=32G"
            ]

        self.slurm_instructions = device_instructions




if __name__ == "__main__":
    launcher = ResampleLauncher()
    launcher.run()