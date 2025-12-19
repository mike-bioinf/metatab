import sys
from metatab.cli.programs.resample.parser import parse_args
from metatab.cli.programs.resample.main_programs.main_tune import main_tune
from metatab.cli.programs.resample.main_programs.main_default import main_default
from metatab.cli.programs.resample.main_programs.main_ensemble import main_ensemble
from metatab.cli.programs.resample.main_programs.main_family_ensemble import main_family_ensemble



def main():
    pars = vars(parse_args(sys.argv[1:]))
    if pars["estimator_mode"] == "default":
        main_default(pars)
    elif pars["estimator_mode"] == "tune":
        main_tune(pars)
    elif pars["estimator_mode"] == "ensemble":
        main_ensemble(pars)
    elif pars["estimator_mode"] == "family_ensemble":
        main_family_ensemble(pars)



if __name__ == "__main__":
    main()