from xmlrpc.client import boolean
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from plotly.subplots import make_subplots

class data_management():
    def __init__(self, csv_path, label):
        """This constructor preprocesses the data
        and plots a scatter matrix for the dataset.

        Args:
            csv_path (data): data path
            label (_type_): takes a target
        """
        # Path for data
        self.data = pd.read_csv(csv_path)

        # Printing the diamension, missing values, number of duplicates and data type of the dataset.
        print('Dataframe dimensions: ----->', self.data.shape)
        print(f'Missing values in dataset: {self.data.isna().sum().sum()}')
        print(f'Duplicates in dataset: {self.data.duplicated().sum()}, ({np.round(100*self.data.duplicated().sum()/len(self.data),1)}%)')
        print(f'Data types: {self.data.dtypes.unique()}')
        
        # Tries to pop out label from dataset and colorizes the plot.
        # If there is no label then the plot will run without label.
        try:
            self.label = self.data.pop(label)

            fig = px.scatter_matrix(self.data,
            color = self.label,
            title = f"Scatter Matrix of Wine dataset colored by {self.label}"
            )
            fig.show()
        except Exception:
            fig = px.scatter_matrix(self.data,
            title = "Scatter Matrix of Wine dataset without label"
            )
            fig.show()


    def pca(self, n_components: int, use_pca: bool, plot: bool):  
        """This constructor takes the data and standardizes it
        then dimension reduces the data with PCA.

        Args:
            n_components (int): number of components the PCA should use
            use_pca (bool): choose if you want to use PCA or not
            plot (bool): choolse if you want to get a scatter plot for PCA
        """
        # Amount of components.
        self.n_components = n_components
        # Checks if user wants to use PCA or not.
        self.use_pca = None
        # Checks if user wants a scatter plot or not.
        self.plot = True
        
        # Standardized the data
        self.stdr_data = StandardScaler().fit_transform(self.data)
        
        # Checks if user wants to use PCA or not. If "True" it will check if user wants a scatter plot or not.
        # If "False" the code will pass.
        try:
            if use_pca == True:
                pca = decomposition.PCA()
                pca.fit(self.stdr_data)

                percentage_var_expl = pca.explained_variance_ / np.sum(pca.explained_variance_)
                cum_var_expl = np.cumsum(percentage_var_expl)

                fig = px.line(
                    cum_var_expl,
                    title="Explained Variance by Components",
                    markers=True
                )

                fig.update_yaxes(title='Cumulative Explained Variance')
                fig.update_xaxes(title='Number of Components')
                fig.update_layout(showlegend=False)
                fig.show()

                pca = decomposition.PCA(n_components)
                pca.fit(self.stdr_data)
                self.scores_pca = pca.transform(self.stdr_data)

                if plot == True:
                    try:
                        fig = px.scatter(
                        x = self.scores_pca[:,0],
                        y = self.scores_pca[:,1],
                        title = "PCA Scatter",
                        color = self.label
                        )
                        fig.update_yaxes(title='2st_principal')
                        fig.update_xaxes(title='1st_principal')
                        fig.update_layout(showlegend=False)
                        fig.show()
                    except Exception:
                        fig = px.scatter(
                        x = self.scores_pca[:,0],
                        y = self.scores_pca[:,1],
                        title = "PCA Scatter"
                        )
                        fig.update_yaxes(title='2st_principal')
                        fig.update_xaxes(title='1st_principal')
                        fig.update_layout(showlegend=False)
                        fig.show()
                elif plot == False:
                    pass
            elif use_pca == False:
                pass
        except Exception:
            pass
    
    def tsne(self, n_components: int, perplexity: int, learning_rate, plot: bool):
        """This constructor takes PCA data or standardized data based on users
        input and then fits it to t-SNE and afterwards plots the results.

        Args:
            n_components (int): number of components the t-SNE should use
            perplexity (int): in relation to the number of nearby neighbors
            learning_rate (_type_): the gradient update step size is determined by the learning rate
            plot (bool): if you want to get a scatter plot for t-SNE
        """
        # Amount of components.
        self.n_components = n_components
        # Perplexity number
        self.perplexity = perplexity
        # Step size for the learn rate
        self.learning_rate = learning_rate
        # Checks if user wants a scatter plot or not.
        self.plot = None
        
        # Tries to fit and transform the data from PCA(if True) into TSNE,
        # if not found it will use the Standardized data instead.
        try:
            tsne = TSNE(n_components, random_state=24, perplexity = perplexity, learning_rate = learning_rate)
            self.data_edited = tsne.fit_transform(self.scores_pca)
        except Exception:
            tsne = TSNE(n_components, random_state=24, perplexity = perplexity, learning_rate = learning_rate)
            self.data_edited = tsne.fit_transform(self.stdr_data)

        # Checks if user wants a scatter plot or not and tries to use with label.
        if plot == True:
            try:
                fig = px.scatter(
                x = self.data_edited[:,0],
                y = self.data_edited[:,1],
                color = self.label,
                title="t-SNE Scatter",
                labels={
                    "x": "",
                    "y": ""  
                    }
                )
                fig.show()
            except Exception:
                fig = px.scatter(
                x = self.data_edited[:,0],
                y = self.data_edited[:,1],
                title="t-SNE Scatter",
                labels={
                    "x": "",
                    "y": ""  
                    }
                )
                fig.show()
        elif plot == False:
            pass

    def k_means(self):
        """This constructor takes the results from t-SNE
        and fits it into K-Means. Also plots Error Metrics
        to see the right amount of clusters to use. In the
        end it plots the results.
        """
        # calculates all Error Metrics
        elbow_metrics = []
        silhouette_metrics = []
        ch_metrics = []
        db_metrics = []
        min = 2
        max = 8
        all_labels = []
        for i in range(min, max):
            kmeans = KMeans(n_clusters=i, random_state=0).fit(self.data_edited)
            labels = kmeans.labels_
            all_labels.append(labels)

            sse = kmeans.inertia_
            elbow_metrics.append(sse)

            silhouette = silhouette_score(self.data_edited, labels)
            silhouette_metrics.append(silhouette)

            ch_score = calinski_harabasz_score(self.data_edited, labels)
            ch_metrics.append(ch_score)

            db_score = davies_bouldin_score(self.data_edited, labels)
            db_metrics.append(db_score)

        # creates a subplot with all 4 Error Metric
        fig = make_subplots(rows=2, cols=2, start_cell="top-left",
            subplot_titles=("Elbow Metrics", "Silhouette Metrics", "Calinski Harabasz Metrics", "Davies Bouldin Metrics"))

        fig.add_trace(go.Scatter(x = (np.arange(min,max)), y = (elbow_metrics)),
            row=1, col=1)
        fig.add_trace(go.Scatter(x = (np.arange(min, max)), y = (silhouette_metrics)),
            row=1, col=2)
        fig.add_trace(go.Scatter(x = (np.arange(min, max)), y = (ch_metrics)),
            row=2, col=1)
        fig.add_trace(go.Scatter(x = (np.arange(min, max)), y = (db_metrics)),
            row=2, col=2)

        fig.update_layout(showlegend=False)
        fig.show()

        # we choose 3 clusters as we could see it in the subplot with the Error Metrics.
        kmeans_pca_tsne = KMeans(n_clusters = 3, init = 'k-means++', random_state = 24)
        # fits the t-SNE data
        kmeans_pca_tsne.fit(self.data_edited)

        # adds "Component 1", "Component 2" and the cluster number "Mean Shift Score" to mean_shift
        data_segm_pca_kmeans = pd.concat([self.data.reset_index(drop = True), pd.DataFrame(self.data_edited)], axis = 1)
        data_segm_pca_kmeans.columns.values[-2: ] = ['Component 1', 'Component 2']
        data_segm_pca_kmeans['K-means Score'] = kmeans_pca_tsne.labels_

        # gives all the cluster numbers a title
        data_segm_pca_kmeans['Segment'] = data_segm_pca_kmeans['K-means Score'].map({
            0:'first',
            1:'second',
            2:'third'
        })

        print(data_segm_pca_kmeans.head())

        # plots K-Means cluster
        fig = px.scatter(
            x = data_segm_pca_kmeans['Component 2'],
            y = data_segm_pca_kmeans['Component 1'],
            color = data_segm_pca_kmeans['Segment'],
            title = 'Clusters by TSNE Components with K-means'
        )
        fig.update_yaxes(title='Component 1')
        fig.update_xaxes(title='Component 2')
        fig.update_layout(showlegend=True)
        fig.show()

        fig = px.scatter_matrix(self.data,
            color = data_segm_pca_kmeans['Segment'],
            title = "Scatter Matrix of Wine dataset with clusters"
        )
        fig.show()
