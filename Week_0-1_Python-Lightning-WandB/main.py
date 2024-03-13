import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="./configs", config_name='config')
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(cfg.model.name)

main()