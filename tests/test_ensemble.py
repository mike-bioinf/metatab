import json
from metalearning.utils import BestMetaStrategyParams
from ensemble.configuration import UserEnsembleConfiguration, CollectionUserEnsembleConfiguration



def factory_user_ensemble_configuration(**kwargs) -> UserEnsembleConfiguration:
    uec = UserEnsembleConfiguration(
        name="name",
        algo="meta",
        n_members=1,
        estimator="random_forest",
        preprocessing="base",
        tune_space="c0",
        meta_strategy="best",
        meta_strategy_params=None,
        early_stop_on_validation_set=False
    )
    
    uec_dict = uec.model_dump()

    for k, v in kwargs.items():
        uec_dict[k] = v
    
    return UserEnsembleConfiguration(**uec_dict)



def test_user_ensemble_configuration_serialization(tmp_path_factory):
    folder = tmp_path_factory.mktemp("generic_folder")
    
    uec = factory_user_ensemble_configuration(
        meta_strategy_params=BestMetaStrategyParams(n_candidate_points=1)
    )
    
    json_string = uec.model_dump_json()
    file = folder / "example.json"
    
    with open(file, "w") as f:
        f.write(json_string)

    with open(file, "r") as f:
        json_loaded = json.load(f)

    uec2 = UserEnsembleConfiguration(**json_loaded)
    assert isinstance(uec2.meta_strategy_params, BestMetaStrategyParams), "Problem in meta_strategy_params serialization."



def test_collection_user_ensemble_configuration_predefined_creation_works():
    collection = CollectionUserEnsembleConfiguration.create_predefined_collection("cpu_meta_2")
    estimators = [conf.estimator for conf in collection.configurations]
    assert "tabpfn" not in estimators, "CPU predefined creation include gpu estimators."
    n_members = [conf.n_members for conf in collection.configurations]
    assert len(set(n_members)) == 1, "Wrong n_members result from predefined creation."
    assert n_members[0] == 2, "Wrong n_members result from predefined creation."
    algo = [conf.algo for conf in collection.configurations]
    assert len(set(algo)) == 1, "Wrong algo result from predefined creation."
    assert algo[0] == "meta", "Wrong algo result from predefined creation."


# test for no error
def test_collection_user_ensemble_configuration_serialization_workflow(tmp_path_factory):
    folder = tmp_path_factory.mktemp("generic_folder")
    json_file = folder / "example.json"
    collection = CollectionUserEnsembleConfiguration.create_predefined_collection("all_meta_2")
    collection.dump_json(json_file)
    collection_reloaded = CollectionUserEnsembleConfiguration.load_json(json_file)