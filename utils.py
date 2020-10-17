import pandas as pd
import numpy as np
import string
import h5py
from random import seed, randint
import time
from cuml import TSNE, UMAP
from openTSNE.sklearn import TSNE as TSNE_cpu
import umap


def get_letter_label(select_by_nspikes):
    """
    This function set an alphabetic encoding to the neurons present in the tetrode

    Parameters
    ----------
    select_by_nspikes: pd.DataFrame with the needed tetrodes for this code to run

    Returns
    -------
    neuron_letters: key dictionary to transform numbers to letters in neurons tags
    """

    alf = string.ascii_lowercase  # lowercase alphabetic characters list
    # A list containing as much labels as neurons are there in the dataset
    # ([aa, ... ,ba, ... ,ca, ... , da])
    labels = ["a" + letter for letter in alf] + ["b" + letter for letter in alf] + \
             ["c" + letter for letter in alf] + ["d" + letter for letter in alf]
    neuron_letters = dict()  # key dictionary for fast replacement of neuron labels
    for k, i in enumerate(select_by_nspikes.neuron):
        neuron_letters[i] = labels[k]
    return neuron_letters


def get_register(random_tetrode, neuron_letters):
    """
    Get the registers from the hdf5 database file

    Parameters
    ----------
    random_tetrode: pd.DataFrame with selected neurons metadata (animal, tetrode, date, neuron_name)
    neuron_letters: dict to asign a unique alphabetic label for the neurons present

    Returns
    -------
    dataset: (pd.DataFrame) containing the tetrode registers with aditional metadata
    """

    columns = []  # Samples columns names
    for j in range(200):
        columns.append(str(j))
    columns2 = columns + random_tetrode.columns.to_list()[:-1]  #
    dataset = pd.DataFrame(columns=columns2)

    with h5py.File('./tesis/Data/RandomSelection.hdf5', 'r') as f:
        for i in random_tetrode.neuron:
            # For each neuron in the random_tetrode, the corresponding signals are retained
            data = pd.DataFrame(np.asarray(f[str(i)][:]), columns=columns)
            # The remaining 'features' are retained from the tetrode dataset
            data['animal'] = random_tetrode.loc[random_tetrode.neuron == i, 'animal']
            data['tetrode'] = random_tetrode.loc[random_tetrode.neuron == i, 'tetrode']
            data['date'] = random_tetrode.loc[random_tetrode.neuron == i, 'date']
            data['neuron'] = i
            # Append to a bigger dataset that has all the neurons from the tetrode:
            dataset = dataset.append(data, ignore_index=True)
    # Replace the neuron numbers with the letters key dictionary
    dataset = dataset.replace({'neuron': neuron_letters})
    # print(dataset['neuron'][0:20])
    return dataset


def synthetic_tetrode_sampler(num_neu, dataset, case_seed):
    """
    Sample random neurons without replacement, to form a tetrode
    Parameters
    ----------
    num_neu: (int) number of neurons in the tetrode
    dataset: (int) dataset of selected neurons to sample from
    case_seed: seed used for sampling the neurons

    Returns
    -------
    random_tetrode: (pd.DataFrame) contains the names of the neurons in the tetrode
    """

    repete = True
    # Guarantee reproducibility in the sampling of the neurons for each tetrode
    seed(case_seed)
    while repete:
        # Sample random neurons without replacement, to form a tetrode
        tetrode_seed = randint(0, 100000)
        random_tetrode = dataset.sample(n=num_neu, replace=False, random_state=tetrode_seed)
        # Drop the ones which come from the same tetrode of the same animal
        # (force them to be from different tetrodes)
        random_tetrode.drop_duplicates(subset=['animal', 'tetrode'])
        # If there were neurons from the same tetrode (very low probability) resample.
        repete = (num_neu != len(random_tetrode))
        # This reseeding guarantees that each time the code is runned, same results are obtained
        seed(tetrode_seed)
    return random_tetrode


def generate_projection(registers, algorithm, implementation, random_init):
    """
    This function generates a projection of the signals to a two dimensional space either
     using tsne or umap algorithms, and returns the projections and computing time.

    Parameters
    ----------
    registers: (pd.DataFrame), datapoints to reduce
    algorithm: (str)  algorithm name, possible: umap, tsne
    implementation: (str) whether ot use gpu or cpu implementation
    random_init: (int) random number seed to the models

    Returns
    -------
    projection: (np.array) number of signals x 2 features.
    """

    print('Calculando proyección')
    assert algorithm in ['tsne', 'umap'], 'Not a valid algorithm name, possible options tsne or umap'
    if algorithm == 'tsne' and implementation == 'gpu':
        model = TSNE(perplexity=30, learning_rate=200, metric='euclidean', random_state=random_init,
                     n_iter=1000, min_grad_norm=1e-07, learning_rate_method=None, verbose=1,
                     method='barnes_hut')
        start_time = time.time()
        projection = model.fit_transform(registers.values)
        exectime = time.time() - start_time

    elif algorithm == 'umap' and implementation == 'gpu':
        model = UMAP(n_neighbors=15, min_dist=0.1, learning_rate=1)
        start_time = time.time()
        projection = model.fit_transform(registers.values)
        exectime = time.time() - start_time

    elif algorithm == 'tsne' and implementation == 'cpu':
        model = TSNE_cpu(perplexity=30, learning_rate=200, metric='euclidean', random_state=random_init,
                         n_iter=1000, initialization='random', min_grad_norm=1e-07, n_jobs=1,
                         negative_gradient_method='bh')
        start_time = time.time()
        projection = model.fit_transform(registers.values)
        exectime = time.time() - start_time
    else:
        model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', learning_rate=1,
                          random_state=random_init)
        start_time = time.time()
        projection = model.fit_transform(registers.values)  # Resulting projection
        exectime = time.time() - start_time

    return projection, exectime


def projections_random_tet(dataset, num_neu, randomness, neuron_letters, case_seed, case,
                           algorithm, implementation):
    """
    Generates a synthetic tetrode sampling neurons randomly from the dataset. Then the projection of the new
    tetrode is generated with the initialization depending on the experiment kind.
    Parameters
    ----------
    dataset: (pd.DataFrame) Possible candidates for the tetrodes
    num_neu: (int) Number of neurons sampled for the new tetrode
    randomness: (str) whether to seed the random initialization of the algorithm ('random_tet')
        or leave it random ('random_random')
    neuron_letters: (dict) dictionary containing the keys for transforming neurons numbers to letters
    case_seed: (int) seed to use in the tetrode sampling to guarantee reproducibility
    case: (int) case number
    algorithm: (str) whether to use umap or tsne
    implementation: (str) whether ot use gpu or cpu implementation

    Returns
    -------
    projections: (dict) contains 'proy' and 'labels' with the projection np.array and the labels tags
    """

    print(f'Processing sample {case} for {num_neu} number of neurons')

    projections = dict()
    # Get the random sampled tetrode
    random_tetrode = synthetic_tetrode_sampler(num_neu, dataset, case_seed)
    # Get the synthetic tetrode from de hdf5 file
    registers = get_register(random_tetrode, neuron_letters)
    # Keep neuron tags
    projections['labels'] = registers.neuron
    # Depending on the execution mode, the algorithms run with a seeded initialization or a random one.
    if randomness == 'random_random':
        # randomRandom means the tetrodes are sampled randomly and the initialization of the algorithm is random
        projections['proy'], projections['time'] = \
            generate_projection(registers.loc[:, '0':'199'], algorithm, implementation, random_init=case_seed)
    elif randomness == 'random_tet':
        # randomTet means the tetrodes are sampled randomly but the algorithm is seeded with zero
        projections['proy'], projections['time'] = \
            generate_projection(registers.loc[:, '0':'199'], algorithm, implementation, random_init=0)
    return projections


def projections_random_state(case, registers, case_seed, algorithm, implementation):
    """
    Computes the projection of the hardcoded registers, with a fixed pseudo-random seed as hyperparameter

    Parameters
    ----------
    case: case number
    registers: dataset with the registers to reduce
    case_seed: seed for each sample
    algorithm: whether to use umap or tsne
    implementation: whether ot use gpu or cpu implementation

    Returns
    -------
    projections: (dict) contains 'proy' and 'labels' with the projection np.array and the labels tags
    """
    projections = dict()
    projections['proy'], projections['time'] = \
        generate_projection(registers.loc[:, '0':'199'], algorithm, implementation, random_init=case_seed)
    return projections
