from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal
from cli.programs.resample.helper import get_resample_iteration_signature

if TYPE_CHECKING:
    from pathlib import Path
    from pandas._libs.missing import NAType
    from estimators.estimators import Estimator




class GeneralManagerEstimatorWorkflowResample:
    '''
    General Estimator workflow interface managining the workflow in the resample iteration.

    Parameters:
        estimator (Estimator):
            Estimator instance.

        pars (dict):
            Parsed and adjusted dict of resample program arguments.
        
        repeat(int | NAType):
            Cross validation repeat. Can be NA when holdout strategy is used.
        
        fold (int):
            Resample iteration. 
            It's the repeat fold in CV or the general iteration in holdout.
    '''
    def __init__(self, estimator: Estimator, pars: dict, repeat: int | NAType, fold: int):
        self._manager_class = self._pick_manager_class(pars["estimator"], pars["estimator_mode"])
        self.manager = self._manager_class(estimator, pars, repeat, fold)
    
    def execute_pre_fit_routine(self):
        self.manager.execute_pre_fit_routine()

    def execute_post_fit_routine(self):
        self.manager.execute_post_fit_routine()
    
    def execute_post_predict_routine(self):
        self.manager.execute_post_predict_routine()
    
    @staticmethod
    def _pick_manager_class(estimator_name: str, estimator_mode: Literal["default", "tune"]):
        match (estimator_name, estimator_mode):
            case ("random_forest", "default"):
                return BaseManagerEstimatorWorkflowResample
            case ("random_forest", "tune"):
                return BaseManagerEstimatorWorkflowResample
            
            case ("xgb", "default"):
                return BaseManagerEstimatorWorkflowResample
            case ("xgb", "tune"):
                return BaseManagerEstimatorWorkflowResample
            case ("es_xgb", "default"):
                return BaseManagerEstimatorWorkflowResample
            case ("es_xgb", "tune"):
                return BaseManagerEstimatorWorkflowResample
            
            case("catboost", "default"):
                return BaseManagerEstimatorWorkflowResample
            case("catboost", "tune"):
                return BaseManagerEstimatorWorkflowResample
            case ("es_catboost", "default"):
                return BaseManagerEstimatorWorkflowResample
            case("es_catboost", "tune"):
                return BaseManagerEstimatorWorkflowResample
            
            case ("lgbm", "default"):
                return BaseManagerEstimatorWorkflowResample
            case("lgbm", "tune"):
                return BaseManagerEstimatorWorkflowResample
            case ("es_lgbm", "default"):
                return BaseManagerEstimatorWorkflowResample
            case ("es_lgbm", "tune"):
                return BaseManagerEstimatorWorkflowResample

            case ("tabpfn", "default"):
                return BaseManagerEstimatorWorkflowResample
            case("tabpfn", "tune"):
                return BaseManagerEstimatorWorkflowResample
            case("autotabpfn", _):
                return ManagerAutoTabPFNWorkflowResample
            case("finetunetabpfn", _):
                return ManagerAesFineTunedTabPFNClassifierWorkflowResample
        
            case _:
                raise ValueError("Unsupported estimator.")



class BaseManagerEstimatorWorkflowResample:
    '''Base Resample Estimator Manager which routines do nothing'''
    def __init__(self, estimator: Estimator, pars: dict, repeat: int | NAType, fold: int):
        self.estimator = estimator
        self.save_estimators: bool = pars["save_estimators"]
        self.output_dir: Path = pars["output_dir"]
        self.repeat = repeat
        self.fold = fold

    def execute_pre_fit_routine(self):
        pass

    def execute_post_fit_routine(self):
        pass
    
    def execute_post_predict_routine(self):
        pass



class ManagerAutoTabPFNWorkflowResample(BaseManagerEstimatorWorkflowResample):
    def execute_pre_fit_routine(self):
        '''
        Create the directory to store the fitted tabpfn models.
        Autogluon save on disk the models and reload them when needed 
        to avoid keeping all of them in memory.
        '''
        iteration_signature = get_resample_iteration_signature(self.repeat, self.fold)
        models_dir = self.output_dir / "autogluon_fitted_models" / f"models_{iteration_signature}"
        self.estimator.set_directory_save_models(models_dir, create_dir=True)

    def execute_post_predict_routine(self):
        '''
        Delete fitted tabpfn models when we do not save the fitted estimators.
        In this scenario these models are useless and we free disk space.
        '''
        if not self.save_estimators:
           self.estimator.delete_save_models_directory()



class ManagerAesFineTunedTabPFNClassifierWorkflowResample(BaseManagerEstimatorWorkflowResample):
    def execute_post_fit_routine(self):
        '''We save the details of the finetune process'''
        folder_stats_finetune = self.output_dir / "stats_finetune"
        os.makedirs(folder_stats_finetune, exist_ok=True)
        iteration_signature = get_resample_iteration_signature(self.repeat, self.fold)
        self.estimator.save_finetune_stats(
            txt_filepath = folder_stats_finetune / f"df_finetune_{iteration_signature}.txt",
            json_filepath = folder_stats_finetune / f"stats_finetune_{iteration_signature}.json"
        )