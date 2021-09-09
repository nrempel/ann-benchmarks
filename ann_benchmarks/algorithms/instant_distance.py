from __future__ import absolute_import
import instant_distance
from ann_benchmarks.algorithms.base import BaseANN


class InstantDistance(BaseANN): 
    def __init__(self, metric, ef_search, ef_construction, ml):
        self._metric = metric
        self._ef_search = ef_search
        self._ef_construction = ef_construction
        self._ml = ml

    def fit(self, X):
        self._points = []

        config = instant_distance.Config()
        config.ef_search = self._ef_search
        config.ef_construction = self._ef_construction
        config.ml = self._ml

        for x in X:
            self._points.append([float(xi) for xi in x.tolist()])
            
        (self._hnsw, ids) = instant_distance.Hnsw.build(self._points, config)
        self._id_map = {ids[i]: i for i in range(len(ids))}

    def query(self, v, n):
        search = instant_distance.Search()
        self._hnsw.search([float(vi) for vi in v.tolist()], search)
        res = [self._id_map[candidate.pid] for candidate in search][:n]
        return res
    
    def __str__(self):
        return 'InstantDistance'