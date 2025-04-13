from options import Options
from tester import Tester
from utils import seed_all

if __name__ == "__main__":
    options = Options()
    opts = options.parse()

    seed_all(opts.random_seed)

    tester = Tester(opts)
    tester.test_qualitative()
    # tester.test_quantitative()
    # tester.test_bound()
    # tester.test_high_derivative()
    # tester.test_grad()
    # tester.test_grad_time()
    # tester.test_inverted_pendulum()
    # tester.test_poly()
