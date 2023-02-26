from xmlrpc.client import boolean
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

class data_management2():
    def __init__(self, csv_path, label: str):
        """This constructor preprocesses the data
        and plots a scatter matrix for the dataset.

        Args:
            csv_path (data): data path
            label (str): takes a target
        """
        # Path for data
        self.data = pd.read_csv(csv_path)

        # Renaming columns because the names was to long.
        self.data.rename(columns = {'Entity' : 'Country',
        'Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)':'alcohol_consumption(L)',
        'GDP per capita, PPP (constant 2017 international $)' : 'gdp_per_cap($)',
        'Population (historical estimates)' : 'population_est'},
        inplace = True)

        # Drop "Code" column because it is not needed.
        self.data.drop("Code", axis=1, inplace=True)
        # Poping out "Country" from dataset if needed later.
        self.country = self.data.pop("Country")
        # Droping all null values.
        self.data = self.data.dropna();
        # Filter all data to year 2015.
        self.data = self.data[self.data['Year'] == 2015]

        # Printing the diamension, missing values, number of duplicates and data type of the dataset.
        print('Dataframe dimensions: ----->', self.data.shape)
        print(f'Missing values in dataset: {self.data.isna().sum().sum()}')
        print(f'Duplicates in dataset: {self.data.duplicated().sum()}, ({np.round(100*self.data.duplicated().sum()/len(self.data),1)}%)')
        print(f'Data types: {self.data.dtypes.unique()}')

        print(self.data.head())

        # Tries to pop out label from dataset and colorizes the plot.
        # If there is no label then the plot will run without label.
        try:
            self.label = self.data.pop(label)

            fig = px.scatter_matrix(self.data,
            dimensions=["Year", "alcohol_consumption(L)", "gdp_per_cap($)", "population_est"],
            color = self.label,
            title = f"Scatter Matrix of Consumption dataset colored by {self.label}"
            )
            fig.show()
        except Exception:
            fig = px.scatter_matrix(self.data,
            dimensions=["Year", "alcohol_consumption(L)", "gdp_per_cap($)", "population_est"],
            title = "Scatter Matrix of Consumption dataset without label"
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
            learning_rate (int/str): the gradient update step size is determined by the learning rate
            plot (bool): choolse if you want to get a scatter plot for t-SNE
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

    def mean_shift(self, quantile: float, bin_seeding: bool):
        """This constructor takes TSNE data and fits it into
        Mean Shift.

        Args:
            quantile (float): the median of all pairwise distances is used
            bin_seeding (bool): setting this option to True will speed up the algorithm because fewer seeds will be initialized
        """
        # median of all pairwise distances
        self.quantile = quantile
        # speeds up the algorithm if True
        self.bin_seeding = bin_seeding


        bandwidth = estimate_bandwidth(self.data_edited, quantile=0.3)
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanshift.fit(self.data_edited)

        # gets the cluster numbers and shows unique
        mean_Shift_Score = meanshift.labels_
        labels_unique = np.unique(mean_Shift_Score)
        n_clusters = len(labels_unique)
        
        print('Estimated number of clusters: ' + str(n_clusters))

        # adds "Component 1", "Component 2" and the cluster number "Mean Shift Score" to mean_shift
        mean_shift = pd.concat([self.data.reset_index(drop = True), pd.DataFrame(self.data_edited)], axis = 1)
        mean_shift.columns.values[-2: ] = ['Component 1', 'Component 2']
        mean_shift['Mean Shift Score'] = mean_Shift_Score

        # gives all the cluster numbers a title
        mean_shift['Segment'] = mean_shift['Mean Shift Score'].map({
            0:'first',
            1:'second',
            2:'third'
        })

        print(mean_shift.head())

        # plots Mean Shift cluster
        fig = px.scatter(
            x = mean_shift['Component 1'],
            y = mean_shift['Component 2'],
            color = mean_shift['Segment'],
            title = 'Clusters by TSNE Components with Mean Shift'
        )
        fig.update_yaxes(title='Component 2')
        fig.update_xaxes(title='Component 1')
        fig.update_layout(showlegend=True)
        fig.show()

        fig = px.scatter_matrix(self.data,
        dimensions=["Year", "alcohol_consumption(L)", "gdp_per_cap($)", "population_est"],
        color = mean_shift['Segment'],
        title = "Scatter Matrix of Consumption dataset with clusters"
        )
        fig.show()