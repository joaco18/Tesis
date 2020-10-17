import argparse
import pickle
import os
import pandas as pd
from random import seed, randint
from multiprocessing import Pool
from contextlib import closing
from itertools import repeat
from .utils import get_letter_label, get_register, synthetic_tetrode_sampler,\
    projections_random_tet, projections_random_state


def random_tetrode_projector(neuron_number, samples, randomness, select_by_nspikes, output_path,
                             algorithm, implementation):
    """
    Create a number (samples) of synthetic tetrodes, with a growing number of neurons from 2 to neuron_number.
    In other words for each number of neurons in the tetrode, generate 'sample' number of synthetic tetrodes.
    Then compute the dimentionality reduction with the selected algorithm and randomness according to the
    experiment name, and finally save the projections.

    Parameters
    ----------
    neuron_number: (int) Maximum number of neurons in the tetrode
    samples: (int) number of samples to reduce in number os neurons by tetrode
    randomness: (str) experiment_name, 'random_state', 'random_tet' or 'random_random'
    select_by_nspikes: (pd.DataFrame) of the selected neurons to be sampled
    output_path: (str) path where the projections are saved
    algorithm: (str) whether to use umap or tsne
    implementation: (str) whether to use gpu or cpu implementation
    """

    # Dictionary with neuron labels as keys to transform them to letters
    neuron_letters = get_letter_label(select_by_nspikes)

    # For each number of neurons in the tetrode explored:
    for num_neu in range(2, neuron_number + 1):
        # Seed random number generator
        seed(0)
        # initialize results dictionary
        projections = dict()
        # Generate pseudo-random seeds to guarantee reproducibility:
        case_seeds = [randint(0, 100000) for _ in range(samples)]
        # If the experiment requires to use a pseudo-random seed, use for each sample the specific case seed:
        if (randomness == 'random_random') | (randomness == 'random_tet'):
            # Parallelize samples along cores in a two step process
            # First Step
            half_samples = int(samples / 2)
            case = range(half_samples)
            # Open a pool
            with closing(Pool(10)) as pool:
                # Map the tasks in the cores
                results = pool.starmap(projections_random_tet,
                                       zip(repeat(select_by_nspikes), repeat(num_neu),
                                           repeat(randomness), repeat(neuron_letters),
                                           case_seeds[:half_samples], case,
                                           repeat(algorithm), repeat(implementation)))
                # Close the pool
                pool.terminate()

            case = range(half_samples, samples)
            # Second Step
            # Open a pool
            with closing(Pool(10)) as pool:
                results2 = pool.starmap(projections_random_tet,
                                        zip(repeat(select_by_nspikes), repeat(num_neu),
                                            repeat(randomness), repeat(neuron_letters),
                                            case_seeds[half_samples:], case,
                                            repeat(algorithm), repeat(implementation)))
                pool.terminate()

        else:
            # Get the random sampled tetrode
            random_tetrode = synthetic_tetrode_sampler(num_neu, select_by_nspikes, case_seeds[0])
            # Get the registers from the hdf5
            registers = get_register(random_tetrode, neuron_letters)
            # Keep the neuron labels
            projections['labels'] = registers.neuron
            # Enumeration of sample numbers
            half_samples = int(samples / 2)
            case = range(half_samples)
            # Open a pool
            with closing(Pool(10)) as pool:
                # Map the tasks in the cores
                results = pool.starmap(projections_random_state,
                                       zip(case, repeat(registers), case_seeds[:half_samples],
                                           repeat(algorithm), repeat(implementation)))
                # Close the pool
                pool.terminate()
            case = range(half_samples, samples)
            # Open a pool
            with closing(Pool(10)) as pool:
                results2 = pool.starmap(projections_random_state,
                                        zip(case, repeat(registers), case_seeds[half_samples:],
                                            repeat(algorithm), repeat(implementation)))
                pool.terminate()

        for i in range(len(results)):
            projections[i] = results[i]
        for i in range(len(results), samples):
            projections[i] = results2[i-10]

        # Save projections
        filename = f'{algorithm}_projections_{experiment_name}_{num_neu}.p'
        with open(os.path.join(output_path, filename), 'wb') as fd:
            pickle.dump(projections, fd, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Study dimentionality reduction performance for UMAP or TSNE in CPU implementations')
    parser.add_argument('--input-filename-path', type=str,
                        help='Path where the dataframes in feather format are stored')
    parser.add_argument('--output-path', type=str,
                        help='Path where the resulting projections are saved')
    parser.add_argument('--number-neurons-in-tetrode', type=int, default=30,
                        help='Maximum number of neurons to be in the tetrode')
    parser.add_argument('--number-samples', type=int, default=20,
                        help='Number of samples to run for each number of neurons')
    parser.add_argument('--experiment-name', type=str,
                        choices=['random_state', 'random_tetrode', 'random_random'])
    parser.add_argument('--algorithm', type=str, choices=['tsne', 'umap'])
    args = parser.parse_args()

    select_by_nspikes = pd.read_feather(args.input_filename_path)
    number_of_neurons = args.number_neurons_in_tetrode
    samples = args.number_samples
    experiment_name = args.experiment_name
    output_path = args.output_path
    algorithm = args.algorithm
    implementation = 'cpu'

    random_tetrode_projector(number_of_neurons, samples, experiment_name, select_by_nspikes,
                             output_path, algorithm, implementation)
