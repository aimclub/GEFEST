import os
import shutil
import pickle
from tqdm import tqdm


def design(n_steps: int,
           pop_size: int,
           estimator,
           sampler,
           optimizer):
    """
    Generative design procedure

    :param n_steps: (Int) number of generative design steps
    :param pop_size: (Int) number of samples in population
    :param estimator: (Object) estimator with .estimate() method
    :param sampler: (Object) sampler with .sample() method
    :param optimizer: (Object) optimizer with .optimize() method
    :return: (List[Structure]) designed samples
    """
    path = 'HistoryFiles'

    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    samples = sampler.sample(n_samples=pop_size)

    for i in tqdm(range(n_steps)):
        performance = estimator.estimate(population=samples)

        with open(f'{path}/performance_{i}.pickle', 'wb') as handle:
            pickle.dump(performance, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{path}/population_{i}.pickle', 'wb') as handle:
            pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        samples = optimizer.step(population=samples, performance=performance, n_step=i)

        #extra_samples = sampler.sample(n_samples=int(pop_size/3))
        #samples = samples + extra_samples

    return samples
