from graph import *
from hope import HOPE


def main():
    graph = Graph()
    graph_path = "data/nodes.edgelist"
    embedding_path = "data/embedding.txt"
    graph.create_graph(filename=graph_path)
    dimension = 128

    model = HOPE(graph=graph, dimension=dimension)

    model.save_embeddings(embedding_path)


if __name__ == '__main__':
    main()
