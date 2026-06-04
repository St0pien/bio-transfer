XGB = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
}


RND_FOREST = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 5, 10, 20, 50],
}


SVM = {
    "C": [0.01, 0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
}

LINEAR = {}

MLP = {
    "hidden_layer_sizes": [
        (128,),
        (256,),
        (128, 64),
        (256, 128),
    ],
    "learning_rate_init": [1e-2, 1e-3, 1e-4],
}
