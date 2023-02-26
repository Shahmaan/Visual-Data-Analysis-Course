from trace import Trace
from kmeans import *
from mean_shift import *

def main():

    kmeans_csv_path = "Assignment\Inlämning\wine.csv"
    kmeans = data_management(kmeans_csv_path, "noTarget")
    kmeans.pca(5, True, False)
    kmeans.tsne(2, 10, 'auto', False)
    kmeans.k_means()

    """mean_shift_csv_path = "Assignment\Inlämning\consumption.csv"
    mean_shift = data_management2(mean_shift_csv_path, "Continent")
    mean_shift.pca(2, False, True)
    mean_shift.tsne(2, 30, 'auto', False)
    mean_shift.mean_shift(0.3, True)"""

if __name__ == '__main__':
    main()