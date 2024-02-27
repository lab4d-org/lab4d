import argparse


def get_config():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument("--logname", type=str, default="base")
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"]
    )
    parser.add_argument("--condition_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument(
        "--time_embedding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "linear", "zero"],
    )
    parser.add_argument(
        "--input_embedding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "linear", "identity"],
    )
    parser.add_argument("--save_images_step", type=int, default=1)
    parser.add_argument("--save_model_epoch", type=int, default=50)

    # test
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--drop_cam", action="store_true")
    parser.add_argument("--drop_past", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--suffix", type=str, default="latest")
    config = parser.parse_args()
    return config
