import wandb

sweep_config = {
    "name": "sweep",
    "method": "bayes",
    "metric": {
        "name": "valid_loss",
        "goal": "minimize",
    },  
        "learning_rate": {
            "distribution": "uniform",
            "min": 1e-4,
            "max": 1e-3
        },
        "batch_size": {
            "values": [8, 16, 32]
        },
        # Hyperparameters for seesaw loss (used when loss is "seesaw")
        "seesaw_p": {
            "distribution": "uniform",
            "min": 0.2,
            "max": 1.5
        },
        "seesaw_q": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 3.0
        },
        # Hyperparameters specific to Tversky loss (used when loss is "tversky")
        "tversky_alpha": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0
        },
        "tversky_beta": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0
        },
        # Hyperparameters specific to weighted sum of dice and seesaw loss (used when loss is "weightedsum_diceseesaw")
        "dice_weight": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0
        },
        "seesaw_weight": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0
        },
        # Hyperparameters specific to focal loss (used when loss is "focal" or "focalcomboloss")
        "focal_alpha": {
            "distribution": "uniform",
            "min": 50,
            "max": 100
        },
        "focal_gamma": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 2.0
        }
    }