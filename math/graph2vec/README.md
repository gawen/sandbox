# Graph distance represented in a small vector space

Given a set of points in a graph, given a set of weight vertices between these points. I define a distance function D(a,b), a & b points of the graph, such as the sum of weights of the vertices of the shortest path between a and b.

The goal is to find vector(s) for every point of the graph which encodes the distance information, so that getting the distance between two points is deported to matrix calculus.

`nmf.py` is a sandbox Python script which uses [non-negative matrix factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) to try to solve this problem.

