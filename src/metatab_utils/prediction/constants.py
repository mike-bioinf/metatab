MANDATORY_COLUMNS = [
    "dataset", 
    "classification_setting", 
    "classes", 
    "classes_counts", 
    "test_labels", 
    "pred_labels", 
    "pred_proba"
]


MUST_COLUMNS_TO_PARSE = [
    "classes", 
    "classes_counts", 
    "test_labels", 
    "pred_labels", 
    "pred_proba"
]


# These columns can be absent
OPTIONAL_COLUMNS_TO_PARSE = [
    "explained_variance_ratio"
]


PERFORMANCE_METRICS = [
    "recall", 
    "precision", 
    "f1", 
    "accuracy",
    "ap",
    "auc"
]