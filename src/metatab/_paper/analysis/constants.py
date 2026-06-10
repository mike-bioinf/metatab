MAP_CLASSIFIERS = {
    "tabpfn": "TabPFN",
    "es_xgb": "ES-XGBoost",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "es_lgbm": "ES-LightGBM",
    "catboost": "CatBoost",
    "es_catboost": "ES-CatBoost",
    "random_forest": "RandomForest",
    "extra_trees": "ExtraTrees",
    "autogluon": "AutoGluon",
    "tabm": "TabM",
    "realmlp": "RealMLP"
}

CATEGORIES_CLASSIFIERS = [
    "ES-LightGBM", 
    "ES-XGBoost", 
    "ES-CatBoost", 
    "RealMLP", 
    "TabM",
    "LightGBM", 
    "XGBoost", 
    "ExtraTrees", 
    "RandomForest", 
    "CatBoost", 
    "TabPFN", 
    "AutoGluon", 
    "CFE"
]

PALETTE_CLASSIFIERS = {
    "ES-LightGBM": "#1f77b4",
    "ES-XGBoost": "#ff7f0e",
    "ES-CatBoost": "#7ec7cf",
    "XGBoost": "#bcbd22",
    "LightGBM": "#2ad52a",
    "CatBoost": "#8c564b",
    "TabM": "#e377c2",
    "RealMLP": "#7f7f7f",
    "ExtraTrees": "#966cbd",
    "RandomForest": "#d62728",
    "TabPFN": "#000000",
    "CFE":"#166E38",
    "AutoGluon": "#580D6D"
}

PALETTE_CROSS_CLASSIFIERS = {
    "Default--CFE": "#166E38",
    "HPO--CFE": "#166E38",
    "Ensemble--CFE": "#166E38",
    "AutoGluon": "#580D6D",
}

SHAPE_REGIMES = {
    "Default": "o",
    "HPO": "s",
    "Ensemble": "X",
    "Meta-HPO": "D",
    "Meta-Ensemble": "^",
    "AutoGluon": "v"
}

PALETTE_REGIMES = {
    "Default": "#1f77b4",
    "Ensemble":  "#2ca02c",
    "HPO": "#ff7f0e",
    "Meta-HPO": "#d62728",
    "Meta-Ensemble": "#B3B345",
    "AutoGluon": "#580D6D",
}


CRC_DATASETS= [
    "Feng_2015",
    "Gupta_2019",
    "Hannigan_2017",
    "Thomas_2018",
    "Thomas_2018b",
    "Vogtmann_2016",
    "Wirbel_2018",
    "Yachida_2019",
    "Yu_2015",
    "Zeller_2014",
]

# Inflammatory bowel disease (IBD)
IBD_DATASETS=[
    "Franzosa_2019",
    "Lloyd-Price_2019",
    "Nielsen_2014",
]

# Nonalcoholic fatty liver disease (NAFLD)
NAFLD_DATASETS=[
    "Hoyles_2018",
    "Loomba_2017",
]

# Crohn's disease (CD)
CD_DATASETS=[
    "He_2017",
    "Lewis_2015",
]

# Type 2 diabetes (T2D)
T2D_DATASETS=[
    "Karlsson_2013",
    "Qin_2012"
]
 
DISEASE_MAP = {
    **{ds:"CRC" for ds in CRC_DATASETS},
    **{ds:"IBD" for ds in IBD_DATASETS},
    **{ds:"NAFLD" for ds in NAFLD_DATASETS},
    **{ds:"CD" for ds in CD_DATASETS},
    **{ds:"T2D" for ds in T2D_DATASETS},
}

MULTI_DISEASE_DATASETS = list(DISEASE_MAP.keys())

# Singleton diseases (only one dataset available)
SINGLETON_DATASETS=[
    "Bedarf_2017",
    "CastroNaller_2015",
    "Chng_2016",
    "Ghensi_2019m",
    "Ghensi_2019p",
    "Jie_2017",
    "Kushugolova_2018",
    "LeChatelier_2013",
    "Li_2017",
    "Qin_2014",
    "Wen_2017",
    "Ye_2018",
    "Zhang_2015",
]

NON_GUT_DATASETS = [
    "CastroNaller_2015",
    "Ghensi_2019m",
    "Ghensi_2019p",
]