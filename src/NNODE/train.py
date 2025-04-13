from options import Options
from trainer import Trainer
from utils import seed_all

if __name__ == "__main__":
    options = Options()
    opts = options.parse()

    seed_all(opts.random_seed)

    trainer = Trainer(opts)
    trainer.train()
