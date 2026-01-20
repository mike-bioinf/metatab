import argparse

from metatab.cli.parser import (
    make_base_parser,
    make_extra_base_parser,
    make_base_resample_parser,
    make_resample_seed_parser,
    make_cv_parser, 
    make_holdout_parser, 
    make_tune_parser,
    make_ensemble_parser,
    make_family_ensemble_parser,
    make_autogluon_parser
)



def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    # first layer resample mode: cv or holdout
    sub_resample_mode = p.add_subparsers(required=True, title="Resample Mode", description="Valid subcommands")
    p_holdout = sub_resample_mode.add_parser("holdout", help="Holdout resample")
    p_cv = sub_resample_mode.add_parser("cv", help="Cross-validation resample")

    # second layer estimator mode: default, tune, ensemble, family-ensemble or autogluon
    sub_holdout_estimator_mode = p_holdout.add_subparsers(required=True, title="Estimator Mode", description="Valid subcommands")
    
    p_holdout_default = sub_holdout_estimator_mode.add_parser(
        "default", 
        parents=[make_base_parser(), make_extra_base_parser(), make_base_resample_parser(), make_resample_seed_parser(), make_holdout_parser()]
    )    
    p_holdout_tune = sub_holdout_estimator_mode.add_parser(
        "tune", 
        parents=[make_base_parser(), make_extra_base_parser(), make_base_resample_parser(), make_resample_seed_parser(), make_holdout_parser(), make_tune_parser()]
    )
    p_holdout_ensemble = sub_holdout_estimator_mode.add_parser(
        "ensemble", 
        parents=[make_base_parser(), make_extra_base_parser(), make_base_resample_parser(), make_resample_seed_parser(), make_holdout_parser(), make_ensemble_parser()]
    )
    p_holdout_family_ensemble = sub_holdout_estimator_mode.add_parser(
        "family-ensemble", 
        parents=[make_base_parser(), make_base_resample_parser(), make_resample_seed_parser(), make_holdout_parser(), make_family_ensemble_parser()]
    )
    p_holdout_autogluon = sub_holdout_estimator_mode.add_parser(
        "autogluon", 
        parents=[make_base_parser(), make_base_resample_parser(), make_holdout_parser(), make_autogluon_parser()]
    )

    p_holdout_default.set_defaults(splitting_mode="holdout", estimator_mode="default")
    p_holdout_tune.set_defaults(splitting_mode="holdout", estimator_mode="tune")
    p_holdout_ensemble.set_defaults(splitting_mode="holdout", estimator_mode="ensemble")
    p_holdout_family_ensemble.set_defaults(splitting_mode="holdout", estimator_mode="family_ensemble")
    p_holdout_autogluon.set_defaults(splitting_mode="holdout", estimator_mode="autogluon")

    sub_cv_estimator_mode = p_cv.add_subparsers(required=True, title="Estimator Mode", description="Valid subcommands")
    
    p_cv_default = sub_cv_estimator_mode.add_parser(
        "default", 
        parents=[make_base_parser(), make_extra_base_parser(), make_base_resample_parser(), make_resample_seed_parser(), make_cv_parser()]
    )
    p_cv_tune = sub_cv_estimator_mode.add_parser(
        "tune", 
        parents=[make_base_parser(), make_extra_base_parser(), make_base_resample_parser(), make_resample_seed_parser(), make_cv_parser(), make_tune_parser()]
    )
    p_cv_ensemble = sub_cv_estimator_mode.add_parser(
        "ensemble", 
        parents=[make_base_parser(), make_extra_base_parser(), make_base_resample_parser(), make_resample_seed_parser(), make_cv_parser(), make_ensemble_parser()]
    )
    p_cv_family_ensemble = sub_cv_estimator_mode.add_parser(
        "family-ensemble", 
        parents=[make_base_parser(), make_base_resample_parser(), make_resample_seed_parser(), make_cv_parser(), make_family_ensemble_parser()]
    )
    p_cv_autogluon = sub_cv_estimator_mode.add_parser(
        "autogluon",
        parents=[make_base_parser(), make_base_resample_parser(), make_cv_parser(), make_autogluon_parser()]
    )

    p_cv_default.set_defaults(splitting_mode="cv", estimator_mode="default")
    p_cv_tune.set_defaults(splitting_mode="cv", estimator_mode="tune")
    p_cv_ensemble.set_defaults(splitting_mode="cv", estimator_mode="ensemble")
    p_cv_family_ensemble.set_defaults(splitting_mode="cv", estimator_mode="family_ensemble")
    p_cv_autogluon.set_defaults(splitting_mode="cv", estimator_mode="autogluon")

    return p.parse_args(args)