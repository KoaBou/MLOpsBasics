from omegaconf import OmegaConf
import hydra

@hydra.main(config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.preferences.user)

if __name__ == "__main__":
    main()