from __future__ import absolute_import
import instant_distance
from ann_benchmarks.algorithms.base import BaseANN


class InstantDistance(BaseANN):
    def __init__(self, metric):
        self._search_k = None
        self._metric = metric

        print("init")
        print("metric", metric)


    def fit(self, X):
        # self._annoy = annoy.AnnoyIndex(X.shape[1], metric=self._metric)
        print("fit")
        points = []
        for i, x in enumerate(X):
            points.append([float(xi) for xi in x.tolist()])
        self._hnsw = instant_distance.Hnsw.build(points, instant_distance.Config())

    def set_query_arguments(self, search_k):
        self._search_k = search_k
        print("set_query_arguments", search_k)

    def query(self, v, n):
        print("query", v, n)
        search = instant_distance.Search()
        self._hnsw.search([float(vi) for vi in v.tolist()], search)
        return search[:n]
        # return self._annoy.get_nns_by_vector(v.tolist(), n, self._search_k)

    def __str__(self):
        return 'Annoy(n_trees=%d, search_k=%d)' % (self._n_trees,
                                                   self._search_k)
