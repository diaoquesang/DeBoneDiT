from dataclasses import dataclass


@dataclass
class config():
    use_server = True
    test_epoch_interval = 10
    image_size = 1024
    r = 8

    batch_size = 8
    epoch_number = 2000
    initial_learning_rate = 1e-4
    milestones = [1200, 1500, 1800]
    num_train_timesteps = 1000

    num_infer_timesteps = 50
    ema = True
    prediction_type = "noise"
    noise_correction = False
    s = 1
