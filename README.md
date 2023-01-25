# UMAP

Uniform Manifold Approximation and Projection (UMAP) is an algorithm for dimensionality reduction which tries to preserve the local distances that the data has in the original dimension. like t-SNE and Isomap. This repository provides an implementation of UMAP dimensionality reduction algorithm in Python from scratch. 

Main components of the algorithm are as follows

- Finding K-nearest neighbors with their corresponding distances in the original data
- Using K-nearest neighbor graph, constructing a fuzzy simplicial set.
- Initializing the low-dimensional embeddings with a logical start
- Optimizing the low-dimensional embeddings using the fuzzy simplicial set constructed from original data.

## Dependencies

```bash
cd PROJECT_PATH
pip -r requirements.txt
```

## Example Usage
```python
import UMAP
X, y = load_digits(return_X_y=True)
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
y_pred = mp.fit_transform(X)
fig, ax = plt.subplots(figsize=(12, 10))
color = y
plt.scatter(y_pred[:,0], y_pred[:,1], c=color, cmap="Spectral", s=2)
plt.setp(ax, xticks=[], yticks=[])
plt.title("Data embedded into two dimensions", fontsize=18)
plt.savefig("./reduced_data.png")
```

## Example Figures

<p align="center">
  <img width="460" height="175" src="./images/mnist.PNG">
</p>

<p align="center">
  <img width="460" height="175" src="./images/load_digits.PNG">
</p>