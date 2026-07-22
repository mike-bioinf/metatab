from metatab._paper.analysis.contour.contour import (
    plot_continuous_continuous,
    plot_nominal_continuous,
    plot_nominal_nominal,
    plot_nominal_ordinal,
    plot_ordinal_ordinal
)


MAP_ESTIMATOR = {
    "catboost": {
        "plot_function": plot_continuous_continuous,
        "x_column": "learning_rate",
        "y_column": "l2_leaf_reg",
        "log_x": True,
        "log_y": True,
    },

    "lgbm": {
        "plot_function": plot_continuous_continuous,
        "x_column": "learning_rate",
        "y_column": "reg_alpha",
        "log_x": True,
        "log_y": True,
    },

    "es_lgbm": {
        "plot_function": plot_continuous_continuous,
        "x_column": "reg_alpha",
        "y_column": "min_split_gain",
        "log_x": True,
        "log_y": True,
    },

    "xgb": {
        "plot_function": plot_continuous_continuous,
        "x_column": "learning_rate",
        "y_column": "gamma",
        "log_x": True,
        "log_y": True,
    },

    "realmlp": {
        "plot_function": plot_continuous_continuous,
        "x_column": "plr_sigma",
        "y_column": "lr",
        "log_x": True,
        "log_y": True,
    },

    "tabpfn": {
        "plot_function": plot_nominal_ordinal,
        "nominal_col": "model_path",
        "ordinal_col": "softmax_temperature",
        "cluster_by": "fix"
    },

    "random_forest": {
        "plot_function": plot_ordinal_ordinal,
        "x_column": "max_features",
        "y_column": "min_samples_leaf",
        "numeric": False,
        "type_plot": "contour",
    },

    "extra_trees": {
        "plot_function": plot_ordinal_ordinal,
        "x_column": "max_features",
        "y_column": "min_samples_leaf",
        "numeric": False,
        "type_plot": "contour",
    },

    "es_xgb": {
        "plot_function": plot_nominal_continuous,
        "nominal_col": "subsample",
        "continuous_col": "reg_alpha",
        "log_cont": True,
        "sigma": 1,
        "cluster_by": "fix"
    },

    "es_catboost": {
        "plot_function": plot_continuous_continuous,
        "x_column": "learning_rate",
        "y_column": "l2_leaf_reg",
        "log_x": True,
        "log_y": True,
    },

    "tabm": {
        "plot_function": plot_nominal_continuous,
        "nominal_col": "batch_size",
        "continuous_col": "lr",
        "log_cont": True,
        "sigma": 1
    },
}