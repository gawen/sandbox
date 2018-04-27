from sklearn.manifold import *
import numpy as np
import sys
import os
import string
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

class Node:
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.edges = {}
        self.idx = None
        self.vector = None

    def __repr__(self):
        b = ['<node {}'.format(self.name)]

        if self.idx is not None:
            b.append(' #{}'.format(self.idx))

        if self.vector is not None:
            b.append(' {}'.format(self.vector))

        b.append('>')
        return ''.join(b)

    def __getitem__(self, k):
        return self.edges[k]

    def __iter__(self):
        return iter(self.edges)

    def __setitem__(self, k, v):
        self.edges[k] = v
        k.edges[self] = v

    def __delitem__(self, k):
        del self.edges[k]
        del k.edges[self]

    def __lt__(self, other):
        return self.name < other.name

    def get_shortest_path(self, end):
        # We always need to visit the start
        nodes_to_visit = {self}
        visited_nodes = set()
        # Distance from start to start is 0
        distance_from_start = {self: 0}
        tentative_parents = {}

        while nodes_to_visit:
            # The next node should be the one with the smallest weight
            current = min(
                [(distance_from_start[node], node) for node in nodes_to_visit]
            )[1]

            # The end was reached
            if current == end:
                break

            nodes_to_visit.discard(current)
            visited_nodes.add(current)

            unvisited_neighbours = set(current).difference(visited_nodes)
            for neighbour in unvisited_neighbours:
                neighbour_distance = distance_from_start[current] + \
                                     current[neighbour]
                if neighbour_distance < distance_from_start.get(neighbour,
                                                                float('inf')):
                    distance_from_start[neighbour] = neighbour_distance
                    tentative_parents[neighbour] = current
                    nodes_to_visit.add(neighbour)

        if end not in tentative_parents:
            return None
        cursor = end
        path = []
        while cursor:
            path.append(cursor)
            cursor = tentative_parents.get(cursor)
        return list(reversed(path))

    def dist(self, other):
        if self == other:
            return 0

        path = self.get_shortest_path(other)

        dist = 0

        prev = None
        for node in path:
            if prev is not None:
                dist += prev[node]
            prev = node

        return dist

    def nodes(self):
        to_visit = set(self)
        seen = set()

        while to_visit:
            visit = to_visit.pop()
            seen.add(visit)
            to_visit.update(set(visit).difference(seen))

        seen = list(seen)
        seen.sort(key=lambda x: x.name)
        return seen

    def vertices(self):
        to_visit = set(self)
        seen = set()

        while to_visit:
            visit = to_visit.pop()
            seen.add(visit)
            new_to_visit = set(visit).difference(seen)
            for i in new_to_visit:
                yield visit, i, visit[i]

            to_visit.update(new_to_visit)

    def dist_mat(self):
        nodes = self.nodes()

        n = np.zeros((len(nodes), len(nodes)))
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                n[i,j]=ni.dist(nj)

        return n

    def mat_similarity(self):
        return np.linalg.inv(self.dist_mat())

    def assign_vectors(self, components=None):
        M = self.dist_mat_mds(components)

        nodes = list(self.nodes())
        for idx, node in enumerate(nodes):
            node.idx = idx

            node.vector = M[idx]

    def vector_dist(self, other):
        v = self.vector - other.vector
        return np.sqrt(v.dot(v.transpose()))

    def dist_err(self, other):
        return self.dist(other) - self.vector_dist(other)

    def mat_dist_err(self):
        nodes = self.nodes()

        n = np.zeros((len(nodes), len(nodes)))
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                n[i,j]=ni.dist_err(nj)

        return n

    def mat_dist_err_ratio(self):
        nodes = self.nodes()

        n = np.zeros((len(nodes), len(nodes)))
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                d = ni.dist(nj)
                dv = ni.vector_dist(nj)
                n[i,j]=d/dv if d != 0 != dv else 0

        return n


    def mat_dist_err_sq(self):
        m = self.mat_dist_err()
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i,j]=m[i,j]**2

        #m = np.log(m)
        return m

    def mat_dist_err_abs_log(self):
        m = self.mat_dist_err()
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i,j]=np.log(np.abs(m[i,j]))

        return m

    def dimension(self):
        def any_similar_point(M):
            for i in range(M.shape[0]):
                for j in range(i+1, M.shape[0]):
                    vi = M[i, :]
                    vj = M[j, :]

                    if np.allclose(vi, vj):
                        return True

            return False

        dim = 1
        while True:
            M = self.dist_mat_kernelpca(dim)

            if not any_similar_point(M):
                break

            dim += 1

        return dim

    def plot_dimension_by_error(self, err):
        def gen():
            for i in tqdm(err):
                yield i, self.dimension_mds(i)[0]

        M = np.mat(list(gen()))
        plt.figure(figsize=(20, 20))
        plt.plot(M[:, 0], M[:, 1])
        plt.show()

    def dist_mat_mds(self, components=None):
        if components is None:
            components = int(os.environ.get('COMPONENTS', '2'))

        model = MDS(components)
        return model.fit_transform(self.dist_mat())

    def plot_mds(self, components=None):
        M = self.dist_mat_mds(components)
        nodes = self.nodes()

        # display
        plt.figure(figsize=(20, 20))
        colors = plt.cm.get_cmap('hsv', len(nodes))
        colors = [colors(i) for i in range(len(nodes))]
        plt.scatter(M[:, 0], M[:, 1], marker='o', c=colors)

        def annotate(label, x, y):
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points', ha='right', va='bottom',
                #bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                #arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
            )

        for node, x, y in zip(nodes, M[:, 0], M[:, 1]):
            node.vector = [x, y]
            annotate('{}'.format(node.name), x, y)

        for s, e, v in self.vertices():
            plt.plot([s.vector[0], e.vector[0]], [s.vector[1], e.vector[1]])

        plt.show()

    def dimension_mds(self, max_err=.1, start_dim=1):
        dim = start_dim
        direction = None
        while True:
            print('projecting to dimension {}...'.format(dim))
            self.assign_vectors(dim)
            E = self.mat_dist_err_ratio()
            err = np.max(E)

            print(E)
            print(err, max_err)

            if err < max_err:
                if direction in (None, -1):
                    dim -= 1
                    direction = -1
                else:
                    dim -= 1
                    break
            else:
                if direction in (None, +1):
                    dim += 1
                    direction = +1
                else:
                    dim += 1
                    break
        return dim

def generate_loops(l):
    P = [Node(name) for _, name in zip(range(l), string.ascii_uppercase)]

    prev = P[0]
    for i in P[1:]:
        prev[i] = 1
        prev = i

    #P[-1][P[0]] = 1
    #P[0][P[l//2]]=1
    #P[l//4][P[3*l//4]] = 1
    #P[2][P[-3]] = 1
    return P[0]

def generate_random_graph(nodes=5, vertices=2):
    P = set(Node(str(i)) for i in range(nodes))

    for p in P:
        neigh = set([p])

        for _ in range(vertices):
            n = random.choice(list(P.difference(neigh)))
            p[n] = 1
            neigh = neigh.difference(n)

    return P.pop()

COUNTER = 0
def generate_tree(depth=3, child=2):
    def new_node(d):
        global COUNTER

        n = Node(str(COUNTER))
        COUNTER += 1

        if d < depth:
            for _ in range(child):
                n[new_node(d+1)] = 1

        return n

    return new_node(0)

def main():
    a = generate_tree(3, 3)
    #a = generate_random_graph(16)
    N = a.nodes()
    print(len(N))

    print(a.dist_mat_mds(2))
    a.plot_mds(2)

    if True:
        dim = a.dimension_mds(1, len(N))
        print(dim)
        a.assign_vectors(dim)
        x, y = random.choice(N), random.choice(N)
        print(x.dist(y), x.vector_dist(y), 10**np.max(a.mat_dist_err_abs_log()))


if __name__ == '__main__':
    main()
