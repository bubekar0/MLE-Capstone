import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import copy
from sklearn.cluster import KMeans
FIG_WIDTH = 18
FIG_HEIGHT = 8
LABEL_SIZE = 20
TICKS_SIZE = 16
RANDOM_STATE = 403

def pca_features(given_PCA, cols, coeffs, feat_count=5):
    '''For a given PCA component, return the main features and weights.
    Args:    given_PCA(Integer): index of the PCA component to analyze
             cols (List): the feature names
             coeffs(PCA Object): contains the feature coefficients for each PCA component.
             feat_count(Integer, default 5): Number of features to return, from largest to smallest
                                             in an absolute value sense.
    Returns: (DataFrame): List of Features and Weights with most impact on the given_PCA
    '''
    ipca_df = pd.DataFrame(coeffs.components_).iloc[given_PCA]
    komponent = pd.DataFrame(list(zip(cols, ipca_df)), columns=['Original Data Feature', 'Weights'])
    komponent['Absolute']=komponent['Weights'].apply(lambda x: np.abs(x))
    return komponent.sort_values('Absolute', ascending=False).head(feat_count)

def show_pca(coeffs, given_PCA, cols, feat_count=10):
    '''For a given PCA component, out of N_COMPONENTS=355, show its makeup in terms of feature weights,
    as well as the cumulative variance retained by all PCA components up to and including this one.
    The feature weights (positive or negative), are sorted by absolute value and displayed interactively.
    A second plot showing the cumulative variance retained vs. number of components is shown alongside.
    Args:    coeffs(PCA Object): contains the feature coefficients for each PCA component.
             given_PCA(integer, range 0 to N_COMPONENTS=355): The PCA component to analyze
             cols (List): Contains the feature names
             feat_count(integer, default 10): Number of features to display, from largest to smallest
                                                      in an absolute value sense.
    Returns: None
    '''

    show_features = pca_features(given_PCA, cols, coeffs, feat_count)
    f = plt.figure(figsize=(.65*FIG_WIDTH, 2*FIG_HEIGHT))
    ax1 = f.add_subplot(211)
    ax1 = sns.barplot(data=show_features, y='Original Data Feature', x='Weights', palette="Blues_d")
    ax1.set(xlabel=None)
    ax1.set(ylabel=None)


    _ = plt.xlabel('Feature Weights in PCA Component Space', fontsize=LABEL_SIZE)
    _ = plt.ylabel('Original Data Features', fontsize=LABEL_SIZE)
    _ = plt.xticks(fontsize=TICKS_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    ax1.set_title("PCA_{} Makeup @ {}% Cumulative Retained Variance".\
                 format(given_PCA, round(100 * coeffs.explained_variance_ratio_.cumsum()[given_PCA]     ) ),\
                 fontsize=LABEL_SIZE)
    ax2 = f.add_subplot(212)
    _ = plt.title("Cumulative Retained Variance", fontsize=LABEL_SIZE)
    _ = plt.xlabel('PCA Components', fontsize=LABEL_SIZE)
    _ = plt.ylabel('Retained Variance', fontsize=LABEL_SIZE)
    _ = plt.xticks(fontsize=TICKS_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = plt.yticks(np.arange(0, 1.1, .1), fontsize=TICKS_SIZE)
    _ = plt.plot(np.cumsum(coeffs.explained_variance_ratio_))
    _ = ax2.set_xticks([195], minor=True)
    _ = ax2.xaxis.grid(True, which='minor', linewidth=3)
    _ = ax2.set_yticks([.08, .9], minor=True)
    _ = ax2.yaxis.grid(True, which='minor', linewidth=1)
    _ = plt.text(8,.12,"(PCA_0, 8%)", fontsize=TICKS_SIZE)
    _ = plt.text(115,.92,"(PCA_195, 90%)", fontsize=TICKS_SIZE)
    _ = plt.grid()
    #_ = plt.savefig('images/Figure 7 — PCA Component Features and Retained Variance.jpg', format='jpeg', dpi=1200, bbox_inches='tight')

def show_feats(coeffs, given_PCA, cols, feat_count=10):
    '''For a given PCA component, out of N_COMPONENTS=355, show its makeup in terms of feature weights,
    The feature weights (positive or negative), are sorted by absolute value and displayed interactively.
    Args:    coeffs(PCA Object): contains the feature coefficients for each PCA component.
             given_PCA(integer, range 0 to N_COMPONENTS=355): The PCA component to analyze
             cols (List): Contains the feature names
             feat_count(integer, default 10): Number of features to display, from largest to smallest
                                                      in an absolute value sense.
    Returns: None
    '''

    show_features = pca_features(given_PCA, cols, coeffs, feat_count)
    f = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax1 = plt.suptitle("Use the sliders to change Component and Number of Features", fontsize=LABEL_SIZE)
    ax1 = sns.barplot(data=show_features, y='Original Data Feature', x='Weights', palette="Blues_d")
    ax1.set(xlabel=None)
    ax1.set(ylabel=None)
    _ = plt.ylabel('Original Data Features', fontsize=LABEL_SIZE)
    _ = plt.xlabel('Feature Weights in PCA Component Space', fontsize=LABEL_SIZE)
    ax1.set_title("PCA Component {} Makeup @ {}% Cumulative Retained Variance".\
                 format(given_PCA, round(100 * coeffs.explained_variance_ratio_.cumsum()[given_PCA]     ) ),\
                 fontsize=LABEL_SIZE)

def centroid_inertias(df, kmax=20):
    '''Iterate several cluster densities to compute average centroid distances.
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets.
             kmax(Integer): Maximum number of clusters to try.
    Returns: inertias (List): Average Centroid Distances
    '''
    inertias = []
    for k in range(2, kmax):
        clusters = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        clusters.fit(df)
        inertias.append(clusters.inertia_)
        print('k={}, '.format(k), end='')
    return inertias

def elbow_graph(kmax, inertias):
    '''Produce a graph of k values vs inertias to visualize elbow
    Args:    kmax (Integer): Maximum valu for k (clusters).
             inertias(List): Average Centroid Distances.
    Returns: None
    '''
    f = plt.figure(figsize=(.75*FIG_WIDTH, FIG_HEIGHT))
    x_range = [k for k in range(2, kmax)]
    _ = plt.plot(x_range, inertias)
    _ = plt.title('K-Means Elbow Graph', fontsize=LABEL_SIZE)
    _ = plt.xlabel("Number of Clusters — 'k'", fontsize=LABEL_SIZE)
    _ = plt.xticks(x_range,fontsize=TICKS_SIZE)
    _ = plt.ylabel('Avg. Centroid Distance (Inertia)', fontsize=LABEL_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = plt.grid()
    #_ = plt.savefig('images/Figure 8 — K-Means Elbow Graph.jpg', format='jpeg', dpi=1200, bbox_inches='tight')

def clusterings(col1, col2, tolerance, label1, label2):
    '''Produce three graphs to analyze the clustering differences across two datasets, presumably the
       General Population (col1) and the Customers (col2) datasets.
    Args:    col1, col2(Numpy Array): Cluster labels belonging to each observation of the two datasets.
             tolerance (Float): Percent excess or deficit to color as over(green) or under(red) representation.
             label1, label2 (String): Legend indicators for the two datasets.
    Returns: None
    '''
    df1_vcs = pd.DataFrame(col1).value_counts()
    df2_vcs = pd.DataFrame(col2).value_counts()
    index_1 = df1_vcs.sort_index().index.get_level_values(0)
    values1 = df1_vcs.sort_index().values
    totals1 = df1_vcs.sum()
    index_2 = df2_vcs.sort_index().index.get_level_values(0)
    values2 = df2_vcs.sort_index().values
    totals2 = df2_vcs.sum()
    deltas  = values2/totals2 - values1/totals1
    fatness = 0.4
    farben  = ['red' if (s < - tolerance) else 'green' if (s >  tolerance) else 'gray' for s in deltas]
    fig = plt.figure(figsize = (0.8*FIG_WIDTH, 1.3*FIG_HEIGHT))
    _ = plt.subplots_adjust(wspace=0, hspace=0.05)
    gs = GridSpec(nrows=3, ncols=1, height_ratios=[2, 2, 1])

    ax1 = fig.add_subplot(gs[0, :])
    _ = plt.title(label1 + " and " + label2 + " Clusterings", fontsize=LABEL_SIZE)
    _ = plt.xticks(index_1, fontsize=0)
    _ = plt.ylabel('Observations', fontsize=LABEL_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = plt.bar(index_1, values1, fatness, label=label1)
    _ = plt.bar(index_2+fatness, values2, fatness, label=label2)
    _ = plt.legend(fontsize=0.8*LABEL_SIZE)
    _ = ax1.tick_params(bottom=False)

    ax2 = fig.add_subplot(gs[1, :])
    _ = plt.bar(index_1, values1/totals1, fatness, label=label1)
    _ = plt.bar(index_2+fatness, values2/totals2, fatness, label=label2)
    _ = plt.xticks(index_2, fontsize=0)
    _ = plt.ylabel('Density', fontsize=LABEL_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = ax2.tick_params(bottom=False)

    ax3 = fig.add_subplot(gs[2, :])
    _ = ax3.bar(index_2, deltas, color=farben)
    _ = plt.xlabel(label2 + " Over- or Underrepresentation ", fontsize=LABEL_SIZE)
    _ = plt.xticks(index_2, fontsize=TICKS_SIZE)
    _ = plt.ylabel('Over / Under %', fontsize=LABEL_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = plt.grid()
    _ = plt.tick_params(grid_color='gray', grid_alpha=.5)
    #_ = plt.savefig('images/Figure 9 — Cluster Observations, Densities, and Over- Under-Representation.jpg', format='jpeg', dpi=1200, bbox_inches='tight')

def clusters_heatmap(ccoords, comps, shades):
    '''Produce four separate heatmaps, one on top of the other, for each of the clusters individually.
       The individual heatmaps allow a better optical presentation as each cluster vmin, and vmax values
       are different. This way the heat shading effect is maximized on a cluster per cluster basis.
    Args:    ccoords(DataFrame): Cluster Coordinates.
             comps (Integer): Number of PCA components to consider for the heatmap
             shades (List): Min and Max values for the shading of each individual heatmap
             cols (List): Feature Names
             cutoff(Integer): Number of features per PCA Component to retrieve.
    Returns: None
    '''
    f = plt.figure(figsize=(.8*FIG_WIDTH, .6*FIG_HEIGHT))
    _ = plt.subplots_adjust(wspace=0, hspace=0)
    ax1 = f.add_subplot(411)
    _ = sns.heatmap(ccoords.iloc[0:1,:comps], cmap = 'Greens', vmin=shades[0], vmax=shades[1], cbar=False)
    _ = ax1.set(xticklabels=[])
    _ = plt.yticks(fontsize=1.5*TICKS_SIZE)
    _ = ax1.tick_params(bottom=False)
    ax2 = f.add_subplot(412)
    _ = sns.heatmap(ccoords.iloc[1:2,:comps], cmap = 'Greens', vmin=shades[2], vmax=shades[3], cbar=False)
    _ = ax2.set(xticklabels=[])
    _ = plt.yticks(fontsize=1.5*TICKS_SIZE)
    _ = ax2.tick_params(bottom=False)
    ax3 = f.add_subplot(413)
    _ = sns.heatmap(ccoords.iloc[2:3,:comps], cmap = 'Greens', vmin=shades[4], vmax=shades[5], cbar=False)
    _ = ax3.set(xticklabels=[])
    _ = plt.yticks(fontsize=1.5*TICKS_SIZE)
    _ = ax3.tick_params(bottom=False)
    ax4 = f.add_subplot(414)
    _ = plt.xlabel(" ", fontsize=0)
    _ = plt.xticks(fontsize=1.5*TICKS_SIZE)
    _ = plt.yticks(fontsize=1.5*TICKS_SIZE)
    _ = sns.heatmap(ccoords.iloc[3:4,:comps], cmap = 'Greens', vmin=shades[6], vmax=shades[7], cbar=False)
    #_ = plt.savefig('images/Figure 10 — Cluster PCA Components Heatmap.jpg', format='jpeg', dpi=1200, bbox_inches='tight')

def cluster_PCA_features(cluster_id, pcas, cc_coords, cols, cutoff=5):
    '''Produce the main features linked with the top PCA Components associated with a Cluster.
    Args:    cluster_id(Integer): Cluster ID.
             pcas (Object): The results of a PCA analysis
             cc_coords (DataFrame): The Cluster Centroid coordinates -> cluster_centers_
             cols (List): Feature Names
             cutoff(Integer): Number of features per PCA Component to retrieve.
    Returns: None
    '''
    centroids = cc_coords.iloc[cluster_id]
    main_PCAs = centroids.argsort()[::-1][:cutoff] # From most positive to most negative

    for comp in main_PCAs:
        pcalist = []
        print('    PCA_{:<2}: '.format(comp), end='')
        features = pd.DataFrame(list(zip(cols, pd.DataFrame(pcas.components_).iloc[comp])),\
                        columns = ['Feature', 'Weights'])
        features['Absolute'] = features['Weights'].apply(lambda x: np.abs(x))
        features = features.sort_values('Absolute', ascending=False).head(cutoff)
        for feature, weight in zip(features['Feature'], features['Weights']):
            if ( weight > 0 ):
                pcalist.append(feature + " +" + str(round(100 * weight,2 )))
            else:
                pcalist.append(feature + " " + str(round(100 * weight,2 )))
        print(pcalist)

def examine_clusterings(cc_coords, df1_labels, df2_labels, pcas, cols):
    '''Examine clusterings of two datasets e.g. GenPop vs Custonmers
    and expose the degrees of over and underrepresentation of df_2.
    Also, display the features linked with the PCA components with higher cluster coordinates.
    This assumes that the larger dataset 'df1' was used to fit both PCA and K-Means, and
    that both datasets were subsequently transformed/predicted by PCA and K-Means respectively.
    Args:    cc_coords (DataFrame): The Cluster Centroid coordinates -> cluster_centers_
             df1_labels, df2_labels ((N,) ndarray): Index of the cluster each sample belongs to
             pcas (Object): The results of a PCA analysis fitted to larger dataset
             cols (List): Feature Names - assumed equivalent to both datasets
    Returns: None
    '''
    df1_percents = 100*round(pd.DataFrame(df1_labels).value_counts()/df1_labels.shape[0],4)
    df2_percents = 100*round(pd.DataFrame(df2_labels).value_counts()/df2_labels.shape[0],4)

    print("Cluster, Delta")               # Quick display of the difference in cluster densities across datasets
    print((df2_percents - df1_percents).sort_values(ascending=False))
    surplus_rep = (df2_percents - df1_percents).sort_values(ascending=False)[:2]         # Overrepresent df2
    deficit_rep = (df2_percents - df1_percents).sort_values(ascending=False)[::-1][:2]  # Underrepresent df2

    print('\033[92m' + "")
    for i in range(len(surplus_rep)):
        indice = int(str(surplus_rep.index[i])[1:2])
        print("C_{} Overrepresentation by {}%".format( indice, round(surplus_rep.values[i],2) ))
        cluster_PCA_features(indice, pcas, cc_coords, cols, 5)
    print('\033[91m' + "")
    for i in range(len(deficit_rep)):
        indice = int(str(deficit_rep.index[i])[1:2])
        print("C_{} Underrepresentation by {}%".format( indice, round(deficit_rep.values[i],2) ))
        cluster_PCA_features(indice, pcas, cc_coords, cols, 5)
