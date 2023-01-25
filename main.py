import os
import numpy as np
import scipy.spatial.distance
import matplotlib.pyplot as plt
from bisect import bisect_right
from scipy.stats import multivariate_normal
from typing import Tuple
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.utils import check_random_state
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_digits
from sklearn.neighbors import KDTree
import sklearn.datasets
import scipy.optimize as optimize
from scipy import sparse
import sys
import pynndescent
from typing import Tuple, List, Union, Optional


sys.path.append("../")
from mnist_download_save import download_and_save


class UMAP:
    def __init__(self, n_neighbors: int=10, n_components: int=2, set_operation_ratio: float=1, local_connectivity: float=1, n_epochs: int=100, learning_rate: float=1, min_dist: float=0.01, spread: float=1, repulsion_strength: float=1, neg_sample_rate: int=5, metric: str='euclidean', init_symbol: str='spectral', a: Optional[float]=None, b: Optional[float]=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.set_operation_ratio = set_operation_ratio
        self.local_connectivity = local_connectivity
        self.n_epochs = n_epochs
        self.initial_alpha = learning_rate
        self.min_dist = min_dist
        self.spread = spread
        self.repulsion_strength = repulsion_strength
        self.neg_sample_rate = neg_sample_rate
        self.metric = metric
        self.a = a
        self.b = b
        self.init_symbol = init_symbol


    def __knn_search(self, X: np.ndarray, Q: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        # use fast approximate n neighbors algorithm
        print("knn search")
        index = pynndescent.NNDescent(X)
        index.prepare()
        if Q is None:
            nearest_neighbors, pair_distances = index.query(X, k=self.n_neighbors)
        else:
            nearest_neighbors, pair_distances = index.query(Q, k=self.n_neighbors)
        return nearest_neighbors, pair_distances


    def __smooth_knn_dist(self, dists: np.ndarray, rho: float, bandwidth: float, niter: int) -> float:
        # search algorithm for sigma_i such that
        # sum_exp(-diff/sigma_i) = log2(n_neighbors) 
        SMOOTH_K_TOLERANCE = 1e-5
        low, mid, high = 0, 1, 1e5
        target = np.log2(self.n_neighbors) * bandwidth
        for i in range(niter):
            nz_diffs = dists - rho
            nz_diffs[nz_diffs<0] = 0
            psum = np.sum(np.exp(-nz_diffs/mid))
            if np.abs(psum - target) < SMOOTH_K_TOLERANCE:
                break
            if psum > target:
                high = mid
                mid = (low + high)/2
            else:
                low = mid
                if high == 1e5:
                    mid *= 2
                else:
                    mid = (low + high) / 2
        return mid
        

    def __smooth_knn_dists(self, knn_dists: np.ndarray, niter: int=64, bandwidth: float=1) -> Tuple[np.ndarray, np.ndarray]:
        N = len(knn_dists)
        rhos = np.min(knn_dists, axis=1)
        sigmas = np.zeros(N)
        for i in range(N):
            sigmas[i] = self.__smooth_knn_dist(knn_dists[i, :], rhos[i], bandwidth, niter)
        return sigmas, rhos


    def __compute_membership_strengths(self, knns: np.ndarray, dists: np.ndarray, sigmas: np.ndarray, rhos: np.ndarray) -> np.ndarray:
        N, neigh = dists.shape
        fs_set = np.zeros((N, N))
        for i in range(N): # for all nodes
            for j in range(neigh): # for all neighbors
                fs_set[i, knns[i,j]] = np.exp(-max(0.0, dists[i,j] - rhos[i]) / sigmas[i]) # fill the node's corresponding neighbor
        return fs_set
    

    def __fuzzy_simplicial_set(self, knns: np.ndarray, dists: np.ndarray) -> np.ndarray:
        print("calculating sigmas/rhos")
        sigmas, rhos = self.__smooth_knn_dists(dists)
        print("calculating fs_set")
        fs_set = self.__compute_membership_strengths(knns, dists, sigmas, rhos)
        fs_set = sparse.csr_matrix(fs_set) # convert it to sparse matrix format

        print("symmetrizing fs_set")
        fs_set_T = fs_set.transpose()
        fs_set = fs_set + fs_set_T - fs_set.multiply(fs_set_T)
        return fs_set


    def __spectral_layout(self, graph: np.ndarray) -> np.ndarray:
        N = graph.shape[0]
        # D = np.diag(1/np.sqrt(np.sum(graph, axis=0)))
        # L = np.eye(D.shape[0]) - np.dot(np.dot(D, graph), D)
        diag = 1 / np.sqrt(np.sum(graph, axis=0))
        D = scipy.sparse.diags(diag, [0], shape=(N, N))
        L = scipy.sparse.diags(np.ones(N)) - D.multiply(graph).multiply(D)

        k = self.n_components + 1
        num_lanczos_vectors = max(2*k+1, round(np.sqrt(L.shape[0])))
        # arnoldi algorithm to find eigs
        print("calculating eigs")
        eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(L, k=k, v0=np.ones(L.shape[1]), ncv=num_lanczos_vectors, which='SM', maxiter=L.shape[0]*5, tol=1e-4)
        smallest_eigvecs = eigenvecs[:, 1:k] # from second to k eigenvecs
        return smallest_eigvecs
    

    def __initialize_embedding(self, graph: np.ndarray) -> np.ndarray:
        if self.init_symbol == "spectral":
            print("Spectral embedding")
            embed = self.__spectral_layout(graph)
            expansion = 10 / np.max(embed)
            embed = np.multiply(embed, expansion) + 1/10000 * np.random.rand()
        elif self.init_symbol == "random":
            print("Random embedding")
            embed = 20*np.random.rand(graph.shape[0], self.n_components) - 10
        return embed


    def __f(self, x: np.ndarray) -> np.ndarray:
        y = []
        for i in range(len(x)):
            if(x[i] <= self.min_dist):
                y.append(1)
            else:
                y.append(np.exp(- x[i] + self.min_dist))
        return y


    def __fit_ab(self) -> Tuple[float, float]:
        x = np.linspace(0, self.spread*3, 300)
        dist_low_dim = lambda x, a, b: 1 / (1 + a*x**(2*b))
        p , _ = optimize.curve_fit(dist_low_dim, x, self.__f(x))
        a = p[0]
        b = p[1] 
        return a, b


    def __optimize_embedding(self, graph: np.ndarray, query_embedding: np.ndarray, ref_embedding: np.ndarray, move_ref: bool) -> np.ndarray:
        print("optimizing")
        N = graph.shape[0]
        # graph = sparse.csr_matrix(graph) # convert it to sparse matrix format
        alpha = self.initial_alpha
        self_reference = (query_embedding is ref_embedding)
        if self.a == None and self.b == None:
            a, b = self.__fit_ab()
        else:
            a, b = self.a, self.b
        cx = graph.tocoo() # to loop over sparse matrix
        for e in range(self.n_epochs):
            print("Epoch:", e+1)
            for i,j,v in zip(cx.row, cx.col, cx.data):
                if np.random.rand() <= v: # attractive forces
                    diff = query_embedding[i] - ref_embedding[j]
                    sdist = np.dot(diff, diff)
                    if sdist > 0:
                        delta = (-2 * a * b * sdist**(b-1))/(1 + a*sdist**b)
                    else:
                        delta = 0
                    
                    grad = delta * diff
                    grad[grad > 4] = 4
                    grad[grad < -4] = -4
                    query_embedding[i] = query_embedding[i] + (alpha * grad)
                    if move_ref:
                        ref_embedding[j] -= (alpha*grad)

                    for _ in range(self.neg_sample_rate):
                        k = np.random.randint(0, len(ref_embedding))
                        if i == k and self_reference:
                            continue
                        diff = query_embedding[i] - ref_embedding[k]
                        sdist = np.dot(diff, diff)
                        if sdist > 0:
                            delta = (2 * self.repulsion_strength * b) / ((1/1000 + sdist)*(1 + a * sdist**b))
                        else:
                            delta = 0
                        grad = delta * diff
                        if delta > 0:
                            grad[grad < -4] = -4
                            grad[grad > 4] = 4
                        else:
                            grad = np.ones(len(query_embedding[i])) * 4
                        query_embedding[i] += (alpha * grad)
                        
            alpha = self.initial_alpha*(1 - (e+1)/self.n_epochs)
        return query_embedding


    def fit(self, X: np.ndarray):
        # calculate n_neighbors for each point with distances
        self.X = X
        knns, dists = self.__knn_search(X)
        # create fuzzy simplicial set
        graph = self.__fuzzy_simplicial_set(knns, dists)
        # initialize embedding
        embedding = self.__initialize_embedding(graph)
        # optimize embeddings
        self.embedding_ = self.__optimize_embedding(graph, embedding, embedding, move_ref=True)


    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.embedding_


    def transform(self, Q: np.ndarray) -> np.ndarray:
        knns, dists = self.__knn_search(self.X, Q)
        # create fuzzy simplicial set
        graph = self.__fuzzy_simplicial_set(knns, dists)
        # initialize embedding
        embedding = self.__initialize_embedding(graph)
        # optimize embeddings
        embedding = self.__optimize_embedding(graph, embedding, self.embedding_, move_ref=False)
        return embedding


if __name__ == "__main__":
    # seed
    np.random.seed(0)
    # data read
    '''
    file_name = "mnist.pkl"
    download_and_save(file_to_save=file_name)
    dict = np.load(file_name, allow_pickle=True)
    images = np.concatenate((dict["training_images"], dict["test_images"])).astype(float)
    labels = np.concatenate((dict["training_labels"], dict["test_labels"]))
    N, D = images.shape
    X = images
    y = labels
    '''
    X, y = load_digits(return_X_y=True)

    # hyperparams
    n_components = 2
    n_neighbors = 10
    metric = "euclidean"
    n_epochs = 200
    learning_rate = 1
    init = "spectral"
    min_dist = 0.001
    spread = 1
    set_operation_ratio = 1
    local_connectivity = 1
    repulsion_strength = 1
    neg_sample_rate = 5
    a = None
    b = None


    mp = UMAP(n_components=n_components, 
                n_neighbors=n_neighbors, 
                metric=metric, 
                n_epochs=n_epochs, 
                learning_rate=learning_rate, 
                init_symbol=init, 
                min_dist=min_dist, 
                spread=spread, 
                set_operation_ratio=set_operation_ratio, 
                local_connectivity=local_connectivity, 
                repulsion_strength=repulsion_strength, 
                neg_sample_rate=neg_sample_rate, 
                a=a, 
                b=b
            )
    print("normalizing")
    X /= 255
    y_pred = mp.fit_transform(X)
    fig, ax = plt.subplots(figsize=(12, 10))
    color = y
    plt.scatter(y_pred[:,0], y_pred[:,1], c=color, cmap="Spectral", s=2)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("Data embedded into two dimensions by myUmap", fontsize=18)
    plt.savefig("../umap_trial/reduced_data.png")


