import os
import shutil
import pickle
from tqdm import tqdm


def design(n_steps: int,
           pop_size: int,
           estimator,
           sampler,
           optimizer,
           extra=False):
    """
    Generative design procedure

    :param n_steps: (Int) number of generative design steps
    :param pop_size: (Int) number of samples in population
    :param estimator: (Object) estimator with .estimate() method
    :param sampler: (Object) sampler with .sample() method
    :param optimizer: (Object) optimizer with .optimize() method
    :param extra: (Bool) flag for extra sampling
    :return: (List[Structure]) designed samples
    """

    def _save_res(performance, samples):
        """
        Saving results in pickle format
        :param performance: (List), performance of samples
        :param samples: (List), samples to save
        :return: None
        """
        with open(f'{path}/performance_{i}.pickle', 'wb') as handle:
            pickle.dump(performance, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{path}/population_{i}.pickle', 'wb') as handle:
            pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def _remain_best(performance, samples):
        """
        From current population we remain best only
        :param performance: (List), performance of samples
        :param samples: (List), samples to save
        :return: (Tuple), performance and samples
        """
        # Combination of performance and samples
        perf_samples = list(zip(performance, samples))

        # Sorting with respect to performance
        sorted_pop = sorted(perf_samples, key=lambda x: x[0])[:pop_size]

        performance = [x[0] for x in sorted_pop]
        samples = [x[1] for x in sorted_pop]

        return performance, samples

    path = 'HistoryFiles'

    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    samples = sampler.sample(n_samples=pop_size)

    for i in tqdm(range(n_steps)):
        performance = estimator.estimate(population=samples)

        # Choose best and save the results
        performance, samples = _remain_best(performance, samples)
        print(f'\nBest performance is {performance[0]}')

        _save_res(performance, samples)

        if optimizer:
            samples = optimizer.step(population=samples, performance=performance, n_step=i)

        # Extra sampling if necessary
        # or if optimizer is missing
        if not optimizer or extra:
            if not optimizer:
                samples = sampler.sample(n_samples=pop_size)
            else:
                extra_samples = sampler.sample(n_samples=pop_size)
                samples = samples + extra_samples

    return samples
