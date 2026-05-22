XGB = {
    "n_estimators": [100, 300],
    "max_depth": [4, 8],
    "learning_rate": [0.01, 0.1],
}


RND_FOREST = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10],
}


SVM = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear"],
}

LINEAR = {}


MLP = {
    "hidden_layer_sizes": [(128,), (256,)],
    "learning_rate_init": [1e-3, 1e-4],
}
