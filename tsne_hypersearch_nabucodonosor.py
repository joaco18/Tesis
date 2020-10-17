"""
Code to explore multiple combinations of hyperparameters for t-SNE algorithm
 using openTSNE implementation running the algorithm over all the available CPU cores.
"""

import pandas as pd
import argparse
import pickle
import time
import os
from openTSNE.sklearn import TSNE
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm


def run_all_hyperp_combinations(data, grid_ranges, tetrode_id):
    """
    Run the projections for each of the possible hyperparameters combination.

    Parameters
    ----------
    data: dataframe with datapoints as rowwss and atributes as columns
    grid_ranges: dictionary with the hyperaprameter name as jey and the list
     of possible values as values.
    tetrode_id: id of the analized tetrode

    Returns
    -------
    projections: dictionary with all the resulting projections and each one's metadata

    """

    # Generate all hyperparamenters values possible combinations
    grid = ParameterGrid(grid_ranges)

    # Initialize results dictionary
    projections = dict()

    for combination_number, combination in tqdm(enumerate(grid)):
        # Initialize the algorithm with the especific hyperparameters values
        model = TSNE(perplexity=combination['perplexity'],
                     learning_rate=combination['learning_rate'],
                     metric=combination['metric'],
                     random_state=0,        # guarantee reproducibility
                     n_iter=1000,
                     initialization='random',
                     min_grad_norm=1e-07,
                     n_jobs=-1,             # parallelize over available cores (10)
                     negative_gradient_method='bh')

        # Run dimentionality reduction
        proy = model.fit_transform(data)

        # Save results
        projections[combination_number] = dict()
        projections[combination_number]['proy'] = proy
        projections[combination_number]['params'] = combination  # Conjunto hiperparámetros
        projections[combination_number]['id'] = tetrode_id  #

    return projections


def compute_and_save_all_reductions(input_filename, output_path, number_of_tetrodes):
    """
    Compute and save the dimensionality reduction for each possible combination
    of the explored hyperparameters

    Parameters
    ----------
    input_filename: path to the feather format dataset
    output_path: path to the directory where the resulting projections are saved

    """

    # Start time counter
    start_time = time.time()

    # Load data
    all_data = pd.read_feather(input_filename)

    # Define hypeparameters ranges to explore
    tsne_grid = {'metric': ['euclidean', 'braycurtis', 'chebyshev', 'manhattan'],
                 'learning_rate': [1000, 5000, 10000, 50000, 100000],
                 'perplexity': [160, 240, 320, 480, 640]}

    # For each tetrode
    for tetrode_number in range(number_of_tetrodes):
        print(f'Processing tetrode number: {tetrode_number}')

        # Select the corresponding tetrode data from the dataframe
        selected_tetrode_data = all_data.loc[all_data.loc[:, f'dataset{tetrode_number}'] == 1, '0':'199']

        # Generate projections for all hypeparameters combinations
        tetrode_id = f'Dataset{tetrode_number}'
        tsne_projections = run_all_hyperp_combinations(selected_tetrode_data, tsne_grid, tetrode_id)

        # Save results
        filename = f'tsne_proyections_{tetrode_id}.p'
        with open(os.path.join(output_path, filename), 'wb') as file:
            pickle.dump(tsne_projections, file, protocol=pickle.HIGHEST_PROTOCOL)

    exectime = time.time() - start_time
    print(f'Execution took: {exectime} s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Explore multiple hyperparameters combinations forr TSNE')
    parser.add_argument('--input-filename', type=str,
                        help='Path where the dataframes in feather format are stored')
    parser.add_argument('--output-path', type=str,
                        help='Path where the resulting projections are saved.')
    parser.add_argument('--number-of-tetrodes', type=str,
                        help='Number of tetrodes in the dataset')
    args = parser.parse_args()

    compute_and_save_all_reductions(args.input_filename, args.output_path, args.number_of_tetrodes)
