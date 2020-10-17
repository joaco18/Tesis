import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import chi2
from scipy.stats import iqr
from sklearn.neighbors import DistanceMetric
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def LScore(data, labels, ratio=False):
    """
    Compute L or Lratio score following [Schmitzer et al.]
    Parameters
    ----------
    data: pd.DataFrame, features as columns, datapoint as row
    labels: pd.DataFrame, one column with the labels, datapoints as rows
    ratio: (bool) whether to use L_ratio o L

    Returns
    -------
    L or L_ratio score
    """
    # Merge features and labels
    data.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)
    data = pd.concat(objs=(data, labels), axis=1)
    data.columns = ['f1', 'f2', 'neuron']

    clusters = data.neuron.unique()
    L_score = 0
    if ratio:
        n_per_c = data.iloc[:, -2:].groupby('neuron').agg('count').values  # Número de registros por cluster
    for idx, cluster in enumerate(clusters):
        # Following [Schmitzer et al.] compute mean and covatience matrix for the cluster
        cov = np.cov(data[data.neuron == cluster].loc[:, 'f1':'f2'].values.T)
        mean = data[data.neuron == cluster].loc[:, 'f1':'f2'].mean().values
        mean = mean.reshape(1, 2)
        # Select spikes from other clusters
        not_in_clust = data.loc[data.neuron != cluster, 'f1':'f2']
        d = np.asarray(not_in_clust.values)
        # With the mean and covarience matrix compute mahalanobis distance to the cluster center:
        dist = DistanceMetric.get_metric('mahalanobis', V=cov)  # Se define la métrica
        d = dist.pairwise(d, mean)
        # Apply rest of the formula:
        cluster_score = np.sum((1.0 - chi2.cdf(d, 2)))
        if ratio:
            L_score = L_score + (cluster_score / n_per_c[idx])
        else:
            L_score = L_score + cluster_score
    L_score = L_score / len(clusters)
    return L_score


def intersection_knn(data, labels, ch):
    """
    Determine if there's an intersection between any of the convex hulls of the clusters. If True, compute
    de superposition percentage score. If false, use knn accuracy to measure cluster quality.
    Parameters
    ----------
    data: (pd.dataframe) resulting projection, datapoints as rows, features as columns
    labels: (pd.Series) labels for each example
    ch: (dict) convex hulls obtained with function convex_hull for each cluster

    Returns
    -------
    sps: (float) superposition percentage score
    knn_acc: (float) knn accuracy
    """

    hull_polygon = dict()  # Dictionary with each cluster convex hull poligons
    neurons = labels.unique()  # Neurons in the tetrode
    for i in ch.keys():
        hull_polygon[i] = Polygon(data.loc[ch[i], :].values)
    inter_bool = False  # Intersection flag
    sum_int_ar = 0      # Acumulator of intersection areas
    ch_keys = list(ch.keys())
    hull_polygon_list = []
    for i in range(len(ch_keys)):
        hull_polygon_list.append(hull_polygon[ch_keys[i]])
    all_polygons = MultiPolygon(hull_polygon_list)
    # Compute area of all polygons union
    total_uni_ar = unary_union(all_polygons).area
    for k, i in enumerate(neurons[:-1]):
        for j in neurons[k + 1:]:           # For each polygons pair
            if hull_polygon[i].intersects(hull_polygon[j]):
                inter = hull_polygon[i].intersection(hull_polygon[j])
                inter_area = inter.area
                sum_int_ar = sum_int_ar + inter_area
                inter_bool = True
    if inter_bool:
        sps = sum_int_ar / total_uni_ar  # Superposition Percentage Score
        knn_acc = np.nan  # No need to compute knn accuracy
    else:
        sps = 0  # SPS is cero and compute knn classification
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(data, labels) # Train
        Y = knn.predict(data) # Predict
        knn_acc = accuracy_score(labels, Y)
    return sps, knn_acc


def convex_hull(data, labels):
    """
    Compute the convex hull of each cluster ignoring outliers.

    Parameters
    ----------
    data: (pd.dataframe) resulting projection, datapoints as rows, features as columns
    labels: (pd.Series) labels for each example

    Returns
    -------
    chull: (dict) dictionary with vertices of each cluster convex_hull
    """

    neurons = labels.unique()
    chull = dict()
    for i in neurons:
        # Keep each cluster data and compute mean, cov matrix and mahalobis distances
        datos = data.loc[labels == i, :]
        indices = datos.index
        cov = np.cov(datos.transpose())
        mean = datos.mean().values
        mean = mean.reshape(1, 2)
        dist = DistanceMetric.get_metric('mahalanobis', V=cov)
        di = dist.pairwise(datos.values, mean)
        # Find outliers with intercuartil range.
        qq = iqr(di)
        q1 = np.percentile(di, 25)
        q3 = np.percentile(di, 75)
        inflim = q1 - qq * 1.5
        suplim = q3 + qq * 1.5
        idx = [True] * (datos.shape[0])
        for j, d in enumerate(di):
            if (d < inflim) | (d > suplim):
                idx[j] = False
        datos = datos[idx]
        indices = indices[idx]
        # Compute convex hulll
        hull = ConvexHull(datos, qhull_options=None)
        chull[i] = np.asarray(indices[hull.vertices])
    return chull


def ConvexHullScore(best_models, labels):
    """
    Compute SPS score ond knn accuracy if there was no intersection between clusters
    Parameters
    ----------
    best_models: (dict) for eaach metric: 1,2,3,4: 'best_s','proy','params'.
    labels: (pd.DataFrame) neuron labels

    Returns
    -------
    best_models: same dict with 'sps','knn_acc','c_hull' added for each of the best 4 projections
    """

    for i in best_models.keys():                           # Para cada métrica
        for j in best_models[i].keys():                    # Para cada uno de los 4 mejores modelos
            data = best_models[i][j]['proy'].copy()
            data = pd.DataFrame(data, index=labels.index, columns=('f1', 'f2'))
            # Se calculan las envolventes convexas
            best_models[i][j]['c_hull'] = convex_hull(data, labels)
            best_models[i][j]['sps'], best_models[i][j]['knn_acc'] = \
                intersection_knn(data, labels, best_models[i][j]['c_hull'])
    return best_models