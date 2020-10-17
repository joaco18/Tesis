import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score


def best(score, proy, metric, best_models, params, key):
    """
    Keep the best four hyperparameter combinations according to the value of the metric studied
    Parameters
    ----------
    score: (float) Value of the score
    proy: (np.array) resulting projection
    metric: (str) name of the metric used
    best_models: (dict) dictionary where the results are saved
    params: (dict) hyperparameters used in the projection
    key: (int) number identifying the projection
    """
    k = 1
    if metric in ['S', 'CH']:  # If the metric is better when is higher invert it to minimize:
        score = -score
        k = -1
    if score < best_models[metric]['first']['best_s']:
        best_models[metric]['fourth'] = best_models[metric]['third'].copy()
        best_models[metric]['third'] = best_models[metric]['second'].copy()
        best_models[metric]['second'] = best_models[metric]['first'].copy()
        best_models[metric]['first']['best_s'] = (score.copy()) * k
        best_models[metric]['first']['proy'] = proy.copy()
        best_models[metric]['first']['params'] = params.copy()
        best_models[metric]['first']['model_number'] = key
    elif score < best_models[metric]['second']['best_s']:
        best_models[metric]['fourth'] = best_models[metric]['third'].copy()
        best_models[metric]['third'] = best_models[metric]['second'].copy()
        best_models[metric]['second']['best_s'] = (score.copy()) * k
        best_models[metric]['second']['proy'] = proy.copy()
        best_models[metric]['second']['params'] = params.copy()
        best_models[metric]['second']['model_number'] = key
    elif score < best_models[metric]['third']['best_s']:
        best_models[metric]['fourth'] = best_models[metric]['third'].copy()
        best_models[metric]['third']['best_s'] = (score.copy()) * k
        best_models[metric]['third']['proy'] = proy.copy()
        best_models[metric]['third']['params'] = params.copy()
        best_models[metric]['third']['model_number'] = key
    elif score < best_models[metric]['fourth']['best_s']:
        best_models[metric]['fourth']['best_s'] = (score.copy()) * k
        best_models[metric]['fourth']['proy'] = proy.copy()
        best_models[metric]['fourth']['params'] = params.copy()
        best_models[metric]['fourth']['model_number'] = key
    return best_models


def scores_calculator(projections, labels, filename, output_path):
    """
    Compute S, L, Lratio, CH, DB metrics for each hyperparameter combination.

    Parameters
    ----------
    projections: (dict) resulting from '<algorithm>_hypersearch_nabucodonosor.py'
    labels: (pd.Series) with the labels of the neurons
    filename: (str) base name for the outputfiles
    output_path: path to the directory where the results are saved
    """
    needed_scores = ['L', 'Lratio', 'DB', 'S', 'CH']
    # Dictionary with the best four models for each score considered
    best_models = dict()
    for score in needed_scores:
        best_models[score] = {}
        for model in best_models[score].keys():
            best_models[score][model] = {'best_s': np.inf, 'sps': np.nan, 'knn_acc': np.nan,
                                         'c_hull': np.nan, 'model_number': np.nan}
    results = pd.DataFrame(index=range(len(projections.keys())), columns=needed_scores)
    # For each hyperparameter combination:
    for pn in projections.keys():
        if 'L' in score:
            L = LScore(projections[pn]['proy'], labels)
            best_models = best(L, projections[pn]['proy'], 'L', best_models, projections[pn]['params'], pn)
            results.at[pn, 'L'] = L.copy()
        if 'Lratio' in score:
            Lr = LScore(projections[pn]['proy'], labels, ratio=True)
            best_models = best(Lr, projections[pn]['proy'], 'Lratio', best_models, projections[pn]['params'], pn)
            results.at[pn, 'Lratio'] = Lr.copy()
        if 'DB' in score:
            DB = davies_bouldin_score(projections[pn]['proy'], labels)
            best_models = best(DB, projections[pn]['proy'], 'DB', best_models, projections[pn]['params'], pn)
            results.at[pn, 'DB'] = DB.copy()
        if 'S' in score:
            S = silhouette_score(projections[pn]['proy'], labels, metric='euclidean')
            best_models = best(S, projections[pn]['proy'], 'S', best_models, projections[pn]['params'], pn)
            results.at[pn, 'S'] = S.copy()  # buen criterio arbitrario para empezar
        if 'CH' in score:
            CH = calinski_harabasz_score(projections[pn]['proy'], labels)
            best_models = best(CH, projections[pn]['proy'], 'CH', best_models, projections[pn]['params'], pn)
            results.at[pn, 'CH'] = CH.copy()

    # Compute convex hulls, intersections and knn accuracies
    best_models = ConvexHullScore(best_models, labels)
    # Complete output dataframe with hyperparameters values
    model_params = pd.DataFrame(index=results.index, columns=projections[0]['params'].keys())
    for k in range(len(projections.keys())):
        model_params.loc[k, :] = projections[k]['params']
    full_results = pd.concat(objs=(results, model_params), axis=1)
    # Save
    filename = filename.replace('.p', '')
    full_results.to_feather(os.path.join(output_path, 'scores', f'{filename}_scores'))
    with open(os.path.join(output_path, 'best_models', f'{filename}_best_models.p'), 'wb') as fp:
        pickle.dump(best_models, fp, protocol=pickle.HIGHEST_PROTOCOL)


def best_models_function(path_to_scores, path_to_best_models, algorithm):
    """
    Determine for each tetrode the best projection according to the voting scheme propossed
    Parameters
    ----------
    path_to_scores: path where the scores for each combination of hiperparameters for each tetrode where saved
    path_to_best_models: path where the best models configurations where saved
    algorithm: algorith to analize
    Returns
    -------

    """
    # Resulting scores files for each tetrode
    filenames = os.listdir(path_to_scores)
    # Generate a Dataframe that contains best configuration for each tetrode
    if algorithm == 'tsne':
        columns = ['model_number', 'SPS', 'knn_accuracy', 'learning_rate', 'metric',
                   'perplexity', 'frequency']
    else:
        columns = ['model_number', 'SPS', 'knn_accuracy', 'learning_rate', 'metric',
                   'min_dist', 'n_neighbors', 'frequency']
    best_model_for_dataset = pd.DataFrame(columns=columns)

    for k, filename in enumerate(filenames):
        dataset_name = (filename.split('_')[2]).replace('.p', '')
        # Load best proyections
        with open(os.path.join(path_to_best_models,
                               f'{algorithm}_proyections_{dataset_name}.p_best_models.p'), 'rb') as file:
            results = pickle.load(file)
        metrics = list(results.keys())
        indices = list()
        for metric in metrics:
            indices = indices + [metric]*4
        indices = [indices, [1, 2, 3, 4]*5]
        # Dataframe that contains the scores for each of the 4 best models according to each metric:
        total = pd.DataFrame(index=pd.MultiIndex.from_tuples(list(zip(*indices))),
                             columns=['model_number', 'hyperparameters', 'score', 'SPS', 'knn_accuracy'])
        order_dict = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
        for metric in metrics:
            for order in results[metric].keys():
                order_number = order_dict[order]
                total.loc[metric, order_number]['model_number'] = results[metric][order]['model_number']
                total.loc[metric, order_number]['hyperparameters'] = results[metric][order]['params']
                total.loc[metric, order_number]['score'] = results[metric][order]['best_s']
                total.loc[metric, order_number]['SPS'] = results[metric][order]['sps']
                total.loc[metric, order_number]['knn_accuracy'] = results[metric][order]['knn_acc']
        best_models = {}
        for metric in metrics:
            if np.sum(total.loc[metric, 'SPS'] == 0) >= 2:
                best_models[metric] = np.nanargmax(total.loc[metric, 'knn_accuracy'].values) + 1
            else:
                best_models[metric] = np.nanargmin(total.loc[metric, 'SPS'].values) + 1

        keys = list(total.loc['L', 1].loc['hyperparameters'].keys())
        best_scored = pd.DataFrame(index=metrics, columns=(['model_number', 'score', 'SPS', 'knn_accuracy']+keys))

        for metric_index in best_models.keys():
            best_scored.loc[metric_index, ['model_number', 'score', 'SPS', 'knn_accuracy']] = \
                total.loc[metric_index, best_models[metric_index]].loc[
                ['model_number', 'score', 'SPS', 'knn_accuracy']]
            best_scored.loc[metric_index, keys] = \
                total.loc[metric_index, best_models[metric_index]].loc['hyperparameters']

        # If all the hyperparameters combinations are different:
        if len(best_scored.loc[:, 'model_number'].unique()) == 5:
            if (best_scored.loc[:, 'SPS'] == 0).any():    # If there are no intersectig projecitons
                # Keer the one with higher knn accuracy
                best_model_for_dataset.loc[dataset_name, :] = \
                    best_scored.iloc[np.nanargmax(best_scored.loc[:, 'knn_accuracy'].values), :]
            else:
                best_model_for_dataset.loc[dataset_name, :] = \
                    best_scored.iloc[np.nanargmin(best_scored.loc[:, 'SPS'].values), :]
        elif len(best_scored.loc[:, 'model_number'].unique()) < 5:  # If there's a voting winner
            if len(best_scored.loc[:, 'model_number'].unique()) == 3:
                if ((best_scored.loc[:, 'model_number'].value_counts().iat[0] == 2) &
                        (best_scored.loc[:, 'model_number'].value_counts().iat[1] == 2)):  # Tie case
                    two_best_model_numbers = best_scored.loc[:, 'model_number'].value_counts().index[0:2]
                    sps_A = (best_scored.loc[best_scored.model_number == two_best_model_numbers[0], 'SPS'][0])
                    sps_B = (best_scored.loc[best_scored.model_number == two_best_model_numbers[1], 'SPS'][0])
                    if (sps_A == 0) & (sps_B == 0):  # No superposition case
                        knn_A = best_scored.loc[
                            best_scored.model_number == two_best_model_numbers[0], 'knn_accuracy'][0]
                        knn_B = best_scored.loc[
                            best_scored.model_number == two_best_model_numbers[1], 'knn_accuracy'][0]
                        if knn_A < knn_B:
                            choice = two_best_model_numbers[1]
                        else:
                            choice = two_best_model_numbers[0]
                        best_model_for_dataset.loc[dataset_name, :] = \
                            best_scored.loc[best_scored.model_number == choice, :].iloc[0, :]
                    else:     # All have superposition
                        if sps_A < sps_B:
                            choice = two_best_model_numbers[0]
                        else:
                            choice = two_best_model_numbers[1]
                        best_model_for_dataset.loc[dataset_name, :] = \
                            best_scored.loc[best_scored.model_number == choice, :].iloc[0, :]
            else:  # No tie case
                ind = best_scored.loc[:, 'model_number'].value_counts().index[0]   # Keep most voted
                best_model_for_dataset.loc[dataset_name, :] = \
                    best_scored.loc[best_scored.model_number == ind, :].iloc[0, :]
        best_model_for_dataset.loc[dataset_name, 'frequency'] = \
            best_scored.loc[:, 'model_number'].value_counts().iat[0]/5
    return best_model_for_dataset
