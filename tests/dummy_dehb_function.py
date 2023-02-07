import hydra


@hydra.main(config_path=".", config_name="dummy_dehb_config", version_base="1.1")
def run_dummy(cfg):
    return int(cfg.x > 1)


if __name__ == "__main__":
    run_dummy()
