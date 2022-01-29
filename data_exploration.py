import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import copy
FIG_WIDTH = 18
FIG_HEIGHT = 8
LABEL_SIZE = 20
TICKS_SIZE = 16

def missing_data(df, low_pct=50, high_pct=75):
    '''Display missing data in a given Dataframe both as a distribution and vs % rows affected.
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
             low_pct, high_pct (Integers): Low and High values of percentage missing data to display
    Returns: None
    '''
    missing_distro = df.isnull().sum(axis=1).apply(lambda x:100*round(x/df.shape[1],2))
    pct_missing, pct_rows = empty_rows(df, low_pct, high_pct)

    # Show histograms and cumulative deterioration in row %-age data missing
    f = plt.figure(figsize=(1.2*FIG_WIDTH, 0.8*FIG_HEIGHT))
    ax1 = f.add_subplot(121)
    _ = plt.title("Multimodal Missing Data %-ages", fontsize=LABEL_SIZE)
    _ = plt.xlabel('% Data Missing', fontsize=LABEL_SIZE)
    _ = plt.xticks(fontsize=TICKS_SIZE)
    _ = plt.ylabel('Density', fontsize=LABEL_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = sns.distplot(missing_distro)

    ax2 = f.add_subplot(122)
    _ = plt.title('Missing Data - Ladder Structure', fontsize=LABEL_SIZE)
    _ = plt.xlabel("% Data Missing", fontsize=LABEL_SIZE)
    _ = plt.xticks(fontsize=TICKS_SIZE)
    _ = plt.ylabel("% Rows Missing Data", fontsize=LABEL_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = plt.plot(pct_missing, pct_rows)
    #_ = plt.savefig('images/Figure 1 — Ladder Structure of Missing Data.jpg', format='jpeg', dpi=1200, bbox_inches='tight')

def empty_rows(df, low_pct=50, high_pct=75):
    '''Build a dictionary of the percent of rows missing data as a function of percent data missed.
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
             low_pct, high_pct (Integers): Low and High values of percentage missing data to display
    Returns: x, y lists ready to plot
    '''
    row_pcent = {}
    for r in range(low_pct, high_pct):
        temp_df = df.dropna(axis=0, thresh=int(((100-r)/100)*df.shape[1] + 1))
        row_pcent[r] = round(100*temp_df.isnull().sum().sum()/temp_df.size)
    lists = sorted(row_pcent.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    return x, y

def dropoff(df, cutoff=.3): # Missing >= 25% of data is acceptable
    '''Get a simplistic column drop list based on a cutoff amount of missing data
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
             cutoff (Float): Maximum percentage of missing data considered acceptable
    Returns: None
    '''
    prozent = df.isnull().mean()
    dropcol = list()
    for i in range(len(prozent)):
        if prozent[i] >= cutoff:
            dropcol.append(prozent.index.values[i])
    return dropcol

def label_encode(df, col):
    '''Convert labels to integers in the entries of a given column from a given dataframe.
    A simple encoding ensuring that the missing data is preserved for later imputation.
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
             col (String): Name of a nominal categorical feature to encode
    Returns: None
    '''
    print('\033[93m' + "[{:<30}](BEFORE) ".format(col), end=": ")
    print(df[col].unique())
    df[col] = df[col].astype("category")
    mapping = dict(enumerate(df[col].cat.categories))
    df[col] = df[col].cat.codes
    df.loc[(df[col] == -1 ), col] = np.nan
    print('\033[92m' + "{:<3}( AFTER) ".format(" "), end=": ")
    print(df[col].unique())
    print('\033[94m' + "Mapping ==>> {}".format(mapping))

def disambiguate(df, col, catchall):
    '''Disambiguate phantom duplicate categories and replace catch-all with next bigger
    integer in the feature to preserve the order information.
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
             col (String): Name of a nominal categorical feature to encode
             catchall (String): Default bin for unknwon, not the same as NaN
    Returns: None
    '''
    print('\033[93m' + "[{:<30}](BEFORE) ".format(col), end=": ")
    print(df[col].unique())

    df[col] = df[col].fillna(-1) # Hide NaNs as siloed int
    for i in range(1,10):
        df.loc[(df[col] == str(i) ), col] = int(i)
        df.loc[(df[col] == i ), col] = int(i)
    df.loc[(df[col] == catchall), col] = int(10) # Encode catch-all bin as next in pecking order
    df[col] = df[col].astype("int")
    df.loc[(df[col] == -1 ), col] = np.nan
    print('\033[92m' + "{:<32}( AFTER) ".format(" "), end=": ")
    print(df[col].unique())

def random_selection(col):
    '''Randomly impute a categorical feature while keeping the distribution intact.
    Args:    col (N,) : Any feature from General Population, Customers, or Mailout datasets
    Returns: None
    REFACTOR
             frequencies = [x for x in col.value_counts(normalize=True).values]
    '''
    bin_totals = col.value_counts()
    bin_values = bin_totals.index
    frequencies = [x for x in bin_totals.values/bin_totals.values.sum()]
    col.fillna(col.isnull()*np.random.choice(bin_values, len(col), p=frequencies), inplace=True)

def impute(df, min_bins=3, max_bins=10, max_miss=35, verbose=False):
    '''Carry out the Random Selection imputation of categorical features in a dataframe.
       This function is merely a sanity filter to the random_selection(col) routine.
       The EXCEL spreadsheet 'DIAS Attributes - Values 2017.xls' shows categorical features
       having typically 10 or less slots, with some exceptions.
       We use 50 slots as a threshold to decide if a feature is imputable.
       We also set an arbitrary limit (15%) as the maximum fraction of missing data that
       we are willing to trust this automated methodology to impute.
       Outside these thresholds we intend to handle columns manually, i.e., with visualization.
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
             min_bins(Integer, default 3): Avoids imputing binaries by default
             max_bins(Integer, default 10): Avoid imputing non-standard GenPop features by default
             max_miss(Float, default 35): Avoid imputing features missing more than 35% by default
             verbose (Boolean, default False): Whether or not to display imputation parameters
    Returns: None
    '''
    for feat in df:
        bin_totals = df[feat].value_counts()
        bins_count = bin_totals.count()
        if bins_count >= min_bins and bins_count < max_bins: # Deal only with non-binary categorical features
            isnan_count = df[feat].isnull().sum()  # missing entries to impute
            if isnan_count:   # Deal only with imputable features
                bin_values = bin_totals.index
                frequencies = [x for x in bin_totals.values/bin_totals.values.sum()]
                isnan_pcent = round(100*isnan_count/len(df[feat]))
                if isnan_pcent > max_miss or bin_values.astype(float).any == np.nan: # Skip if missing too much (or if NaN is biggest slot!)
                    if ( verbose ):
                        print('\033[91m' + "{:<28} Found {:>2} slots, NOT IMPUTING {:>2}%".\
                             format(feat, bins_count, isnan_pcent))
                    else:
                        print('\033[91m' + "N", end='')
                else:
                    random_selection(df[feat])
                    if ( verbose ):
                        print('\033[92m' + "{:<28} Found {:>2} slots, imputing {:>2}% with {} and p={}"\
                            .format(feat, bins_count, isnan_pcent, bin_values.tolist(),\
                                [round(foo,2) for foo in frequencies]))
                    else:
                        print('\033[92m' + ".", end='')

def before_and_after(col, label, nanbin=-10, width=FIG_WIDTH, height=FIG_HEIGHT):
    '''Show the effect of imputation without performing it.
    Args:    col (N,) : Any feature from General Population, Customers, or Mailout datasets
             label (String) : Name of the feature, to display on Graph
             nanbin (Integer) : Fake slot to show graphically (in red color) the Nan blank entries
             width, height( Float) : Dimensions of the desired graphs
    Returns: None
    '''
    withnan = copy.deepcopy(col)
    withnan.fillna(value=nanbin, inplace=True)
    w_index = withnan.value_counts().sort_index().index
    w_value = withnan.value_counts().sort_index().values
    numbins = withnan.value_counts().sort_index(ascending=False).count()
    wcolors = ['gray' for x in withnan.value_counts(dropna=False).sort_index().values]
    wcolors[0] = 'red'
    imputed = copy.deepcopy(col)
    imputed.fillna(imputed.isnull()*np.random.choice(imputed.value_counts().index, len(imputed), \
                    p=[x for x in imputed.value_counts().values/imputed.value_counts().values.sum()]), inplace=True)
    icolors = ['black' for x in imputed.value_counts().values]
    i_index = imputed.value_counts().sort_index().index
    i_value = imputed.value_counts().sort_index().values

    f = plt.figure(figsize=(width, height))
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])
    ax1 = f.add_subplot(gs[:, 0])
    _ = plt.bar(w_index, w_value, color=wcolors, label='Empties')

    _ = plt.xticks(fontsize=TICKS_SIZE/2)
    _ = plt.yticks(fontsize=TICKS_SIZE/2)
    _ = plt.legend(fontsize=LABEL_SIZE/2)
    _ = plt.grid()

    ax2 = f.add_subplot(gs[:, 1], sharey=ax1)
    _ = plt.bar(w_index, w_value*0, fill=False) #Hack to get bar-like widths to coincide
    _ = plt.bar(i_index, i_value, color=icolors, label=label + ' (imputed)')

    _ = plt.xticks(fontsize=TICKS_SIZE/2)
    _ = plt.yticks(fontsize=0)
    _ = plt.legend(fontsize=LABEL_SIZE/2)
    _ = ax2.grid()
    #_ = plt.savefig('images/Figure 2 — Random Selection Univariate Imputation.jpg', format='jpeg', dpi=1200, bbox_inches='tight')
    _ = plt.show()


def mean_or_rand_sel(col, label, xrange, nanbin=-10):
    '''Show the data, the result of mean imputation, and the results of our imputation on some integer
       numerical features of interest, particulary KBA13_ANZAHL_PKW.
    Args: col ((N,) shaped structure): Any numerical feature from the datasets.
          xrange( Tuple (min,max)): The low and high range limits for the slots.
          nanbin (Integer) : Fake slot to show graphically (in red color) the Nan blank entries
    Returns: None
    '''
    withnan = copy.deepcopy(col)
    withnan.fillna(value=nanbin, inplace=True)
    meaned = copy.deepcopy(col)
    meaned.fillna(col.mean(), inplace=True)
    imputed = copy.deepcopy(col)
    imputed.fillna(imputed.isnull()*np.random.choice(imputed.value_counts().index, len(imputed), \
            p=[x for x in imputed.value_counts().values/imputed.value_counts().values.sum()]), inplace=True)

    f = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT/5))
    ax3 = f.add_subplot(133)
    _ = sns.distplot(imputed, hist=False, axlabel=False, kde_kws={'clip': xrange})
    _ = plt.legend(labels=['Random Selection'], fontsize=LABEL_SIZE/2)
    _ = plt.xticks(fontsize=TICKS_SIZE/2)
    _ = ax3.set_ylabel('')

    ax1 = f.add_subplot(131, sharex=ax3, sharey=ax3)
    _ = sns.distplot(withnan, hist=False, axlabel=False, kde_kws={'clip': xrange})
    _ = plt.legend(labels=[label], fontsize=LABEL_SIZE/2)
    _ = plt.xticks(fontsize=TICKS_SIZE/2)
    _ = ax1.set_ylabel('')

    ax2 = f.add_subplot(132, sharex=ax3, sharey=ax3)
    _ = sns.distplot(meaned, hist=False, axlabel=False, kde_kws={'clip': xrange})
    _ = plt.legend(labels=['Mean Imputation'], fontsize=LABEL_SIZE/2)
    _ = plt.xticks(fontsize=TICKS_SIZE/2)
    _ = ax2.set_ylabel('')
    #_ = plt.savefig('images/Figure 3 — Mean Imputation vs. Random Selection.jpg', format='jpeg', dpi=1200, bbox_inches='tight')
    _ = plt.show()

def pseudo_numericals(df1, df2, max_cats=50):
    '''Visualize features containing a large number of slots, to spot outliers in two datasets.
    Args:    df1, df2 (DataFrame): General Population, Customers, or Mailout datasets.
             max_cats (Integer): Arbitrary cutoff number of bins beyond which a feature
             becomes eligible for outliers inspection.
    Returns: None
    '''
    for feat in df1:
        bin_counts1 = df1[feat].value_counts().count()
        bin_counts2 = df1[feat].value_counts().count()
        if bin_counts1 > max_cats:
            f = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT/4))
            ax1 = f.add_subplot(121)
            _ = plt.xlabel(" ", fontsize=LABEL_SIZE/2)
            _ = plt.xticks(fontsize=TICKS_SIZE/2)
            _ = plt.yticks(fontsize=TICKS_SIZE/2)
            _ = sns.distplot(df1[feat], bins=bin_counts1)
            ax2 = f.add_subplot(122)
            _ = plt.xlabel(" ", fontsize=LABEL_SIZE/2)
            _ = plt.xticks(fontsize=TICKS_SIZE/2)
            _ = plt.yticks(fontsize=TICKS_SIZE/2)
            _ = sns.distplot(df2[feat], bins=bin_counts2)

def anno_domini_outlier(col, label, ymax):
    '''Fix the left tail outlier of feature GEBURTSJAHR caused by huge percent of the
       individuals reporting birth year zero.
    Args:    col ((N,)) One of genpop['GEBURTSJAHR'], or kunden['GEBURTSJAHR']
             label (String): X-axis label for the bottom graph
             ymax (Integer): Value to clip the outlier display, otherwise the top graph
                             good data part is too small to be visible.
                             ymax = 11000 works well for the General population  dataset
                                     3500 works well for the Customers dataset
                                     1000 works well for the Mailout Train & Test datasets
    Returns: None
    '''
    bincount = len(col.value_counts())
    f = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax1 = f.add_subplot(211)
    _ = ax1.set_ylim(0, ymax)
    _ = plt.xlabel(" ", fontsize=LABEL_SIZE)
    _ = plt.ylabel(" ", fontsize=LABEL_SIZE)
    _ = plt.xticks(fontsize=TICKS_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = plt.bar(col.value_counts().index, col.value_counts().values)
    _ = ax1.legend(labels=['Raw Data with Outlier'], loc=2, fontsize=LABEL_SIZE)

    # Temporarily move the outliers to the "missing" bin for imputing
    col.loc[(col == 0 )] = np.nan
    ax2 = f.add_axes([0.45, 0.6, 0.35, 0.25])
    _ = ax2.set_xlim(1900, 2020)
    _ = plt.xlabel(" ", fontsize=LABEL_SIZE)
    _ = plt.ylabel(" ", fontsize=LABEL_SIZE)
    _ = plt.xticks(fontsize=TICKS_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = plt.bar(col.value_counts().index, col.value_counts().values)
    _ = ax2.legend(labels=['> 1900'], loc=1, fontsize=.75*LABEL_SIZE)

    random_selection(col)
    ax3 = f.add_subplot(212)
    _ = ax3.set_xlim(1890, 2025)
    _ = plt.xlabel(" ", fontsize=LABEL_SIZE)
    _ = plt.ylabel(" ", fontsize=LABEL_SIZE)
    _ = plt.xticks(fontsize=TICKS_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = plt.bar(col.value_counts().index, col.value_counts().values)
    _ = ax3.legend(labels=['Random Selection'], loc=2, fontsize=LABEL_SIZE)
    _ = ax3.set(xlabel=label)
    #_ = plt.savefig('images/Figure 4 — Handling GEBURTSJAHR Outlier.jpg', format='jpeg', dpi=1200, bbox_inches='tight')

def dequantize_right_tail(col, label, xmin=1250, xmax=2300):
    '''Fix the right tail outliers of feature KBA13_ANZAHL_PKW caused by abrupt
       change in granularity of observations.
    Args:    col ((N,)) One of genpop['KBA13_ANZAHL_PKW'], or kunden['KBA13_ANZAHL_PKW']
             label (String): X-axis label for the bottom graph
             xmin, xmax (Integers): The range to impute
    Returns: None
    '''
    bincount = len(col.value_counts())
    f = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax1 = f.add_subplot(211)
    _ = plt.xlabel(" ", fontsize=0)
    _ = plt.ylabel(" ", fontsize=LABEL_SIZE)
    _ = plt.xticks(fontsize=TICKS_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = ax1.set_ylim(0, 0.002)
    _ = sns.distplot(col, bins=bincount)
    _ = ax1.legend(labels=['Outliers'], fontsize=LABEL_SIZE)

    # Clean the tail by dirtying it with NaNs (make eligible to use fillna)
    col.loc[(col > bincount )] = np.nan
    ax2 = f.add_subplot(212)
    _ = plt.xlabel(" ", fontsize=LABEL_SIZE)
    _ = plt.ylabel(" ", fontsize=LABEL_SIZE)
    _ = plt.xticks(fontsize=TICKS_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    # Use average of tail last 100 entries to start decline
    ytop = col.value_counts().sort_index(ascending=False).head(100).mean()
    ybot = 0
    slope = (ybot - ytop)/(xmax - xmin)
    x = np.arange(xmin, xmax)
    y, _ = divmod( slope*(x - xmin) + ytop, 1)
    col.fillna(col.isnull()*np.random.choice( x, len(col), p=y/sum(y) ), inplace=True)
    _ = sns.distplot(col, bins=bincount)
    _ = ax2.set(xlabel=label)
    _ = ax2.legend(labels=['Dequantization'], fontsize=LABEL_SIZE)
    #_ = plt.savefig('images/Figure 5 — Handling KBA13_ANZAHL_PKW Outliers.jpg', format='jpeg', dpi=1200, bbox_inches='tight')

def flag_features(df1, df2, tolerance=0.05):
    '''Loop through all the features of two datasets df2 (Customers) vs df1 (General Population).
    Highlight the degree to which each slot in a Customer's feature either overshoots
    or undershoots the same slot in the General population. We'll consider a Customer's
    feature to overshoot if its density exceeds the General Population's by at least
    the tolerance amount passed as an argument. Undershooting happens when the density
    trails by more than the tolerance.
    Starting with the accepted hypothesis that the customers belong to the general population,
    we'd like to flag features (and their slots) which may lead us to reject the hypothesis.
    Thus, we'll focus on the highlighted features to study how the customers differ from the
    general population and create a profile for the marketing campaigns.
    A difficulty in this comparison arises when the datasets do not consume all the slots in the
    given feature. If the slots consumed differ between datasets, the problem gets complicated.
    Args:    df1(GenPop), df2(Customers) (DataFrame): The dataframes to compare
             tolerance (Float): Percent amount to tolerate before calling it
                                over or undershooting
    Returns: None
    '''
    MAX_SLOT_DISPARITY = 10 # Arbitrary cutoff to avoid too much disparity of slot consumption
    cols = df1.columns.values#[:10]
    for feature in cols:
        idx1 = df1[feature].value_counts().sort_index().index
        idx2 = df2[feature].value_counts().sort_index().index
        df1_rel = df1[feature].value_counts().sort_index().values/df1[feature].value_counts().sum()
        df2_rel = df2[feature].value_counts().sort_index().values/df2[feature].value_counts().sum()

        # Skip cases with excessive disparity in slot consumption across the datasets
        if ( (idx1.shape[0] - idx2.shape[0]) > MAX_SLOT_DISPARITY ):
            print("\n" + '\033[91m' + "Skipping {}".format(feature) + '\33[0m', end='')
        else:
            if (idx1.shape[0] == idx2.shape[0]):
                deltas  = df2_rel - df1_rel
            else: # Try to fix divergent slot. We prefer slot consumption by customers to compare.
                deltas = [pd.DataFrame(df2_rel, index=idx2).loc[s].item() - pd.DataFrame(df1_rel, index=idx1).loc[s].item()\
                          if s in idx1 else pd.DataFrame(df2_rel, index=idx2).loc[s].item() for s in idx2]
            farben = ['red' if (s < - tolerance) else 'green' if (s >  tolerance) else 'gray' for s in deltas]
            if ( (np.array(deltas) < -tolerance).any() or (np.array(deltas) > tolerance).any() ):
                print("\n" + '\33[0m' + '{:>35}: '.format(feature) + '\33[0m', end='')
                for rat in deltas:
                    if ( rat > tolerance ):
                        print('\33[42m' + '{:>7.2f}'.format(rat) + '\33[0m', end='')
                        print('\33[0m' +  ' ' + '\33[0m', end='')
                    elif ( rat < -tolerance):
                        print('\33[41m' + '{:>7.2f}'.format(rat) + '\33[0m', end='')
                        print('\33[0m' +  ' ' + '\33[0m', end='')
                    else:
                        print('\33[0m' + '{:>7.2f}'.format(rat) + '\33[0m', end='')
                        print('\33[0m' +  ' ' + '\33[0m', end='')

def comparator(col1, col2, tolerance, label1, label2, width, height):
    '''Graphically compare a feature across two datasets: col1=df1[feat], col2=df2[feat].
    For this project, we can assume df1=GenPop, and df2=Customers. But this is not necessary.
    Display  the degree to which each slot in a Customer's feature either overshoots
    or undershoots the same slot in the General population. We'll consider a Customer's
    feature to overshoot if its density exceeds the General Population's by at least
    the tolerance amount passed as an argument. Undershooting happens when the density
    trails by more than the tolerance.
    This routine complements flag_features().
    Args:    col1, col2 (Series): The feature across datasets to compare
             tolerance (Float): Percent amount to tolerate before calling it
                                over or undershooting
             label1, label2 (String): Legends to use for df1 and df2 respectively
             width, height (Float): Dimensions of the graph for use with figsize.
    Returns: None
    '''
    MAX_SLOT_DISPARITY = 10 # Arbitrary cutoff to avoid too much disparity of slot consumption
    index_1 = col1.value_counts().sort_index().index
    values1 = col1.value_counts().sort_index().values
    totals1 = col1.value_counts().sum()
    index_2 = col2.value_counts().sort_index().index
    values2 = col2.value_counts().sort_index().values
    totals2 = col2.value_counts().sum()

    # In cases with disparate slot consumption across the datasets, display the top graph alone
    if ( (index_1.shape[0] - index_2.shape[0]) > MAX_SLOT_DISPARITY):
        deltas  = values2/totals2 - values2/totals2
    else:
        if (index_1.shape[0] == index_2.shape[0]):
            deltas  = values2/totals2 - values1/totals1
        else: # Try to fix divergent slot
            deltas = [pd.DataFrame(values2/totals2, index=index_2).loc[s].item() - \
                      pd.DataFrame(values1/totals1, index=index_1).loc[s].item()\
                      if s in index_1 else pd.DataFrame(values2/totals2, index=index_2).loc[s].item() for s in index_2]
        #farben = ['red' if (s < - tolerance) else 'green' if (s >  tolerance) else 'gray' for s in deltas]


    fig = plt.figure(figsize = (width, height))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1.5])
    farben = ['red' if (s < - tolerance) else 'green' if (s >  tolerance) else 'gray' for s in deltas]

    ax1 = fig.add_subplot(gs[0, :])
    _ = plt.title(label2 + " Over- or Undershooting " + label1, fontsize=LABEL_SIZE)
    _ = plt.xticks(index_1, fontsize=TICKS_SIZE)
    _ = plt.ylabel('Observations', fontsize=LABEL_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = ax1.bar(index_1, values1*0, fill=False) #Hack to get bar-like widths to coincide
    _ = ax1.plot(index_1, values1, label=label1)
    _ = ax1.plot(index_2, values2, label=label2)
    _ = plt.grid()
    _ = plt.legend(fontsize=LABEL_SIZE)

    ax2 = fig.add_subplot(gs[1, :])
    _ = ax2.bar(index_2, deltas, color=farben)
    _ = plt.xlabel('Feature Slots', fontsize=LABEL_SIZE)
    _ = plt.xticks(index_2, fontsize=TICKS_SIZE)
    _ = plt.ylabel('Overshoot %', fontsize=LABEL_SIZE)
    _ = plt.yticks(fontsize=TICKS_SIZE)
    _ = plt.grid()
    _ = plt.tick_params(grid_color='gray', grid_alpha=.5)
    #_ = plt.savefig('images/Figure 6 — General Population vs. Customers.jpg', format='jpeg', dpi=1200, bbox_inches='tight')
