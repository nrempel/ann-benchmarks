from __future__ import absolute_import
import instant_distance
from ann_benchmarks.algorithms.base import BaseANN


class InstantDistance(BaseANN):
    def __init__(self, metric):
        self._search_k = None
        self._metric = metric

    def fit(self, X):        
        self._points = []
        for x in X:
            self._points.append([float(xi) for xi in x.tolist()])
            
        (self._hnsw, ids) = instant_distance.Hnsw.build(self._points, instant_distance.Config())
        self._id_map = {ids[i]: i for i in range(len(ids))}

    def set_query_arguments(self, search_k):
        self._search_k = search_k

    def query(self, v, n):
        search = instant_distance.Search()
        self._hnsw.search([float(vi) for vi in v.tolist()], search)
        res = [self._id_map[candidate.pid] for candidate in search][:n]
        return res
    
    def __str__(self):
        return 'Annoy(search_k=%d)' % (self._search_k)
