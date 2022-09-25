import networkx
import numpy
import scipy.sparse.linalg as linalg


class HOPE(object):
    def __init__(self, graph, dimension):
        self.dimension = dimension
        self.graph = graph
        self.train()
        self.vectors = self.vectors()

    def train(self):
        graph = self.graph.Graph
        A = networkx.to_numpy_matrix(graph)

        M_g = numpy.eye(graph.number_of_nodes())
        M_l = numpy.dot(A, A)

        S = numpy.dot(numpy.linalg.inv(M_g), M_l)

        u, sigma_k, vt = linalg.svds(S, k=self.dimension)
        sigma = numpy.diagflat(numpy.sqrt(sigma_k))

        X1 = numpy.dot(u, sigma)
        X2 = numpy.dot(vt.T, sigma)
        self._X = numpy.concatenate((X1, X2), axis=1)

    def vectors(self):
        vectors = {}
        look_back = self.graph.look_back_list

        for index, embedding in enumerate(self._X):
            vectors[look_back[index]] = embedding

        return vectors

    def save_embeddings(self, file):
        file_output = open(file, 'w')
        node_numbers = len(self.vectors.keys())
        file_output.write("{} {}\n".format(node_numbers, self.dimension))

        for node, vector in self.vectors.items():
            file_output.write("{} {}\n".format(
                node, ' '.join([str(x) for x in vector])))
        file_output.close()
