"""
Functions to explore multiple combinations of hyperparameters for t-SNE algorithm
 using openTSNE implementation running the algorithm over all the available CPU cores.
"""
import pandas as pd
import argparse
import pickle
import time
import os
import umap
from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool
from contextlib import closing
from tqdm import tqdm


def read_args():
    parser = argparse.ArgumentParser(
        description='Explore multiple hyperparameters combinations for UMAP')
    parser.add_argument('--input-filename', type=str,
                        help='Path where the dataframes in feather format are stored')
    parser.add_argument('--output-path', type=str,
                        help='Path where the resulting projections are saved')
    args = parser.parse_args()

    return args


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
        model = umap.UMAP(n_neighbors=combination['n_neighbors'],
                          min_dist=combination['min_dist'],
                          metric=combination['metric'],
                          learning_rate=combination['learning_rate'],
                          random_state=0)                    # guarantee reproducibility
        # Run dimentionality reduction
        proy = model.fit_transform(data)

        projections[combination_number] = {}
        projections[combination_number]['proy'] = proy
        projections[combination_number]['params'] = combination
        projections[combination_number]['id'] = tetrode_id

    return projections


def compute_and_save_umap_projections(case):
    """
    Compute and save the dimensionality reduction for each possible combination
    of the explored hyperparameters for one "case" tetrode

    Parameters
    ----------
    case: tetrode number to process
    """
    args = read_args()

    tetrode_id = ('Dataset' + str(case))
    print(f'Processing tetrode number: {tetrode_id}')

    # Load data
    all_data = pd.read_feather(args.input_filename)

    # Define hypeparameters ranges to explore
    base_md = 0.0001
    umap_grid = {'metric': ['euclidean', 'braycurtis', 'chebyshev', 'manhattan'],
                 'learning_rate': [1, 5, 10, 50, 100],
                 'n_neighbors': [10, 20, 40, 60, 80, 100],
                 'min_dist': [base_md, base_md * 10, base_md * 100]}

    # Select the corresponding tetrode data from the dataframe
    selected = all_data.loc[all_data.loc[:, f'dataset{case}'] == 1, '0':'199']

    # Generate projections for all hypeparameters combinations
    umap_projections = run_all_hyperp_combinations(selected, umap_grid, tetrode_id)

    with open(os.path.join(args.output_path, f'umap_proyections_{tetrode_id}.p'), 'wb') as file:
        pickle.dump(umap_projections, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    # Start time counter
    start_time = time.time()

    # Fix number of threath to paralellize over
    num_threads = 20

    # Paralellize tetrodes over cores
    cases = range(20)
    with closing(Pool(num_threads)) as pool:
        pool.map(compute_and_save_umap_projections, cases)
        pool.terminate()

    exectime = time.time() - start_time
    print(f'Execution took: {exectime} s')



















