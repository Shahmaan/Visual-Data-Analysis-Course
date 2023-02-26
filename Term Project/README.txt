PCA
Principal Component Analysis, or PCA, is a dimensionality-reduction method that is
frequently used to reduce the dimensionality of big data sets by reducing a large
collection of variables into a smaller set that retains the majority of the information in the large set.

Naturally, reducing the number of variables in a data collection reduces accuracy,
but the idea in dimensionality reduction is to trade a little accuracy for simplicity.
Because smaller data sets are easier to study and display, and because machine learning
algorithms can analyze data much more easily and quickly without having to deal with superfluous factors.

To summarize, the goal behind PCA is simple: decrease the number of variables in a data collection
while retaining as much information as possible.


TSNE
The t-SNE algorithm approximates the probability distribution of neighbors surrounding each location.
In this context, the term neighbors refers to the collection of points that are closest to each other.

The main parameter controlling the fitting is called perplexity. Perplexity is roughly equivalent to
the number of nearest neighbors considered when matching the original and fitted distributions for each
point. A low perplexity means we care about local scale and focus on the closest other points.
High perplexity takes more of a "big picture" approach.

Because the distributions are based on distance, all of the data must be numerical. You should use binary
encoding or a comparable approach to convert categorical variables to numeric variables. It's also a good
idea to standardize the data so that each variable is on the same scale. This prevents variables having a
wider numerical range from dominating the analysis.


KMEANS
K-means is a clustering algorithm that uses the distance between each data point and a centroid to assign
it to a cluster. The goal is to identify the K number of groups in the dataset. 

It is an iterative process of allocating each data point to a group and gradually clustering data points
based on related features. The goal is to minimize the sum of the distances between the data points and
the cluster centroid in order to determine which group each data point should belong to.

Metrics for KEANS:
As we can see from the metrics plot, all three of the four error metrics indicated that we should implement
three clusters. We can clearly see that there should be three clusters in "Elbow Metrics" (where the elbow is),
"Silhouette Metrics" (where the value is closest to 1.0), and
"Calinski Harabasz Metric" (where the value should be the highest), 
but there are four clusters in "Davies Bouldin Metrics" (where the value should be closest to 0.0).


MEAN SHIFT
Mean Shift is an unsupervised clustering approach that seeks to detect blobs in samples with a smooth density.
It is a centroid-based algorithm that operates by updating centroids candidates to be the mean of the points in
a particular region (also called bandwidth). To generate the final collection of centroids, these candidates are
filtered in a post-processing stage to eliminate near-duplicates. As a result, unlike KMeans, we do not need to
choose the number of clusters manually.

Metrics for MEAN SHIFT:
As previously stated, you do not have to manually set the number of clusters on Mean Shift, and so this model
has no error metric.