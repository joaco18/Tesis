import argparse
import pickle
import os
import pandas as pd
from random import seed, randint
from .utils import get_register, get_letter_label, synthetic_tetrode_sampler,\
    projections_random_tet, projections_random_state


def random_tetrode_projector(neuron_number, samples, experiment_name, dataset,
                             output_path, algorithm, implementation):
    """
    Create a number (samples) of synthetic tetrodes, with a growing number of neurons from 2 to neuron_number.
    In other words for each number of neurons in the tetrode, generate 'sample' number of synthetic tetrodes.
    Then compute the dimentionality reduction with the selected algorithm and randomness according to the
    experiment name, and finally save the projections.

    Parameters
    ----------
    neuron_number: (int) Maximum number of neurons in the tetrode
    samples: (int) number of samples to reduce in number os neurons by tetrode
    experiment_name: (str) experiment_name, 'random_state', 'random_tet' or 'random_random'
    dataset: (pd.DataFrame) of the selected neurons to be sampled
    output_path: (str) path where the projections are saved
    algorithm: (str) whether to use umap or tsne
    implementation: (str) whether to use gpu or cpu implementation
    """

    # Dictionary with neuron labels as keys to transform them to letters
    neuron_letters = get_letter_label(dataset)

    # For each number of neurons in the tetrode explored:
    for num_neu in range(2, neuron_number + 1):
        # Seed random number generator
        seed(0)
        # initialize results dictionary
        projections = dict()
        # Generate pseudo-random seeds to guarantee reproducibility:
        case_seeds = [randint(0, 100000) for _ in range(samples)]
        # If the experiment requires to use a pseudo-random seed, use for each sample the specific case seed:
        if (experiment_name == 'random_random') | (experiment_name == 'random_tet'):
            results = {}
            for case in range(samples):
                results[case] = projections_random_tet(dataset, num_neu, experiment_name, neuron_letters,
                                                       case_seeds[case], case, algorithm, implementation)

        # If not, just use the seed to sample the neurons in the tetrode
        else:
            # Get the random sampled tetrode
            random_tetrode = synthetic_tetrode_sampler(num_neu, dataset, case_seeds[0])
            # Get the registers from the hdf5 file
            registers = get_register(random_tetrode, neuron_letters)
            # Keep the neuron labels
            projections['labels'] = registers.neuron

            results = {}
            for case in range(samples):
                results[case] = projections_random_state(case, registers, case_seeds[case],
                                                         algorithm, implementation)

        for i in range(len(results)):
            projections[i] = results[i]

        filename = f'{algorithm}_projections_{experiment_name}_{num_neu}.p'
        with open(os.path.join(output_path, filename), 'wb') as fd:
            pickle.dump(projections, fd, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Study dimentionality reduction performance for UMAP or TSNE in GPU implementations')
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

    # Read dataset of neurons selected for the experiment
    select_by_nspikes = pd.read_feather(args.input_filename_path)
    number_of_neurons = args.number_neurons_in_tetrode
    samples = args.number_samples
    experiment_name = args.experiment_name
    output_path = args.output_path
    algorithm = args.algorithm
    implementation = 'gpu'
    random_tetrode_projector(number_of_neurons, samples, experiment_name, select_by_nspikes,
                             output_path, algorithm, implementation)
