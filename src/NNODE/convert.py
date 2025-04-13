from options import Options
from converter import Converter
from utils import seed_all

if __name__ == "__main__":
    options = Options()
    opts = options.parse()

    seed_all(opts.random_seed)

    converter = Converter(opts)
    converter.run()
