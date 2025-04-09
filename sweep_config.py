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

config = Config(
    learning_rate=wandb.config.get("learning_rate", 2e-4),
    batch_size=wandb.config.get("batch_size", 16),
    seesaw_p=wandb.config.get("seesaw_p", 0.8),
    seesaw_q=wandb.config.get("seesaw_q", 2.0),
    tversky_alpha=wandb.config.get("tversky_alpha", 0.3),
    tversky_beta=wandb.config.get("tversky_beta", 0.7),
    dice_weight=wandb.config.get("dice_weight", 0.5),
    seesaw_weight=wandb.config.get("seesaw_weight", 0.5),
    focal_alpha=wandb.config.get("focal_alpha", 75),
    focal_gamma=wandb.config.get("focal_gamma", 1),
    focal_beta=wandb.config.get("focal_beta", 2),
)