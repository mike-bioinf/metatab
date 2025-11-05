from __future__ import annotations

import sys
import subprocess
from abc import ABC, abstractmethod
from typing import Literal, TYPE_CHECKING
from metatab_utils.data_loader import DataLoader
from cli.resample.params import parse_args, adjust_splitting_specs_

from metatab_utils.helper_programs import (
    check_fit_resample_args,
    check_tune_algo,
    manage_output_path, 
    adjust_io_paths_,
    adjust_tune_configuration_arg_,
    adjust_early_stopping_rounds_
)

from .constants import (
    CPU_ESTIMATORS, 
    ALL_GPU_ESTIMATORS,
    SINGLE_GPU_ESTIMATORS,
    MULTIPLE_GPU_ESTIMATORS,
    NUMBER_CPUS_ON_PARALLEL_PARTITION,
    NUMBER_CPUS_ON_GPUS_PARTITON
)

if TYPE_CHECKING:
    import pandas as pd




class PolimorphicLauncher:
    '''
    Class with three purposes:
    1) Parse the resample program arguments
    2) Set the desired launcher
    3) Run the launcher
    '''
    def __init__(self, type_launcher: Literal["hard", "soft"]):
        self.type_launcher = type_launcher
        self.launcher = self._get_launcher()

    
    def _get_launcher(self) -> SoftLauncher | HardLauncher:
        launcher_inputs = self._process_program_inputs()

        if self.type_launcher == "hard":
            launcher = HardLauncher(sys.argv[1:], *launcher_inputs)
        elif self.type_launcher == "soft":
            launcher = SoftLauncher(sys.argv[1:], *launcher_inputs)
        else:
            raise ValueError("Unsupported launcher type.")
        
        return launcher
    
    
    def _process_program_inputs(self) -> tuple[dict, pd.DataFrame]:
        '''
        Returns the info coming from the program inputs required by the launcher.
        This require executing oprations that will be done also by the main program, 
        but  we accept the duplication overhead.
        In detail it returns the dict of parsed arguments and the X data.
        '''
        parsed_args = self._parse_program_args()
        X = self._load_data(parsed_args)
        return parsed_args, X


    @staticmethod
    def _parse_program_args() -> None:
        '''Parse, check and adjust the program arguments'''
        pars = vars(parse_args(sys.argv[1:]))
        check_fit_resample_args(pars)
        adjust_io_paths_(pars, "input_data", "output_dir")
        manage_output_path(pars, "output_dir", True)
        adjust_splitting_specs_(pars)
        adjust_tune_configuration_arg_(pars)
        adjust_early_stopping_rounds_(pars)
        check_tune_algo(pars)
        return pars
    

    @staticmethod
    def _load_data(args: dict) -> pd.DataFrame:
        '''Load the input feature space X'''
        dl = DataLoader()
        dl.load(
            mode=args["input_mode"],
            path=args["input_data"],
            target_feature=args["target_feature"],
            load_as="generic"
        )
        return dl.X


    def run(self):
        '''Run the Launcher'''
        self.launcher.run()




class BaseLauncher(ABC):
    '''
    Base call for launchers.

    This class implements the common logic for launching the resample program
    as a Slurm job. It is responsible for:

        - Constructing the Slurm resource instructions based on the selected estimator.
        - Building the final "sbatch" command (including environment setup via the "setup_instruction").
        - Submitting the job with the user-provided program arguments.
    
    The base class never modifies the program arguments. 
    Instead, it delegates any additional feasibility checks to subclasses 
    via the "_check_resample_programs_instructions" method.

    Subclasses:
        - `SoftLauncher`: performs no additional checks and simply launches the job.
        - `HardLauncher`: adds heuristic feasibility checks against cluster
          constraints and raises errors for configurations that are unlikely to
          complete within available resources.

    Parameters:
        args (list[str]): List of raw arguments stored in "sys.argv" without the module name.
        parsed_args (dict): Parsed dict of the program arguments.
        X (pd.DataFrame): X data on which the program is run.
    '''
    def __init__(
        self,
        args: list[str],
        parsed_args: dict, 
        X: pd.DataFrame
    ):
        self.args=args
        self.parsed_args=parsed_args
        self.X=X
        self.slurm_instructions: list[str] = None
        self.setup_instruction: str = (
            "ulimit -u 100000;"
            " source /lustre/home/epasolli/tools/miniconda3/bin/activate finetunetabpfn;"
        )

    
    def run(self):
        self._set_slurm_instructions()
        self._check_resample_programs_instructions()
        self._run_sbatch_job()
        

    @abstractmethod
    def _check_resample_programs_instructions():
        pass
    

    def _set_slurm_instructions(self) -> None:
        fixed_instructions = [
            "--job-name=resample",
            "--output=resample_%j.out",
            "--error=resample_%j.err"
        ]

        estimator = self.parsed_args["estimator"]
        
        if estimator in CPU_ESTIMATORS:
            if self.parsed_args["nthreads"] >= NUMBER_CPUS_ON_PARALLEL_PARTITION:
                raise ValueError(
                    f"The number of user requested threads must be inferior to {NUMBER_CPUS_ON_PARALLEL_PARTITION}."
                )
            device_instructions = [
                "--partition=parallel",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={NUMBER_CPUS_ON_PARALLEL_PARTITION}"
            ]
        
        elif estimator in ALL_GPU_ESTIMATORS:
            if self.parsed_args["nthreads"] > NUMBER_CPUS_ON_GPUS_PARTITON:
                raise ValueError(
                    f"The number of user requested threads must be inferior to {NUMBER_CPUS_ON_GPUS_PARTITON+1}."
                )
            
            n_gpus = 4 if estimator in MULTIPLE_GPU_ESTIMATORS else 1

            device_instructions = [
                "--partition=gpus",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=6",
                f"--gpus-per-node={n_gpus}", # equivalent f"--gres=gpu:{n_gpus}""
                "--mem-per-gpu=32G"
            ]
        
        else:
            raise ValueError("Unsupported estimator")
        
        self.slurm_instructions = fixed_instructions + device_instructions


    def _run_sbatch_job(self) -> None:
        '''
        Create a new process in which the slurm job is launched.
        We spawn the new process via shell (shell=True in the run call)
        since otherwise we get file not found errors for the input program files.
        '''
        if self.slurm_instructions is None:
            raise ValueError("The slurm instructions are not set yet.")

        script_body = (
            self.setup_instruction + 
            " metatab-resample " + 
            " ".join(self._get_quoted_args())
        )

        sbatch_cmd = (
            "sbatch " +
            " ".join(self.slurm_instructions) +
            f' --wrap="{script_body}"'
        )
        
        try:
            _ = subprocess.run(sbatch_cmd, check=True, shell=True)
        except Exception as e:
            raise ValueError(f"The launcher failed to launch the job with error '{e}'")
    

    def _get_quoted_args(self) -> list[str]:
        '''Method that correctly quotes the args passed as str representation of dicts'''
        args = []
        for a in self.args:
            # use single quotes if the double ones are already used for keys
            if "{" in a and "\"" in a:
                args.append(f"'{a}'")
            # use double quotes if the single ones are already used for keys
            elif "{" in a and "'" in a:
                args.append(f"\"\"{a}\"\"")
            else:
                args.append(a)
        return args




class SoftLauncher(BaseLauncher):
    def _check_resample_programs_instructions(self) -> None:
        # do nothing
        return None



# TODO: implement warning on estimator-non_expected_preprocessing combo ??
class HardLauncher(BaseLauncher):
    def _check_resample_programs_instructions(self) -> None:
        tune = self.args["tune"]
        
        # nothing to worry in non-tuning setting
        if not tune:
            return None

        n_rows = self.X.shape[0]
        estimator = self.args["estimator"]
        tune_space = self.parsed_args["tune_configuration"]["configuration"]
        n_tune_iter = self.parsed_args["tune_configuration"]["n_iter"]
        resampling_strategy = self.args["splitting_mode"]
        resampling_specs = self.parsed_args["splitting_specs"]

        if resampling_strategy == "cv":
            n_folds = resampling_specs["n_splits"]
            n_repeats = resampling_specs["n_repeats"]
            n_resample_iter = n_repeats * n_folds
            n_rows_fitted = n_rows / ((n_folds-1)/n_folds)
        else:
            n_resample_iter = resampling_specs["n_splits"]
            n_rows_fitted = n_rows * resampling_specs["train_size"]
        
        if (
            estimator == "catboost" and
            n_rows_fitted > 315 and  # 315 estimated from 350 * 0.9
            tune_space in ["default", "c0", "c3"]  and # the default for now is c0
            n_resample_iter > 20 and 
            n_tune_iter > 50
        ):
            raise ValueError(
                "The program inputs are not compatible with the hardware time constrains." +
                " Try to lower the number of resample iterations, tune iterations"+
                " or lower the number of cv folds or train_size (in holdout)."
                " Keep on mind that this estimatate comes from an euristic."
            )