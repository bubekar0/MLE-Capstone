import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
FIG_WIDTH = 18
FIG_HEIGHT = 8

def empty_rows(df):
    '''Build a dictionary of the percent of rows missing data as a function of percent data missed.
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
    Returns: x, y lists ready to plot
    '''
    row_pcent = {}
    for r in range(50,75):
        temp_df = df.dropna(axis=0, thresh=int(((100-r)/100)*df.shape[1] + 1))
        row_pcent[r] = round(100*temp_df.isnull().sum().sum()/temp_df.size)
    lists = sorted(row_pcent.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    return x, y

def dropoff(df, cutoff=.25): # Missing >= 25% of data is acceptable
    '''Get a simplistic column drop list based on a cutoff amount of missing data
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
             cutoff (float): Maximum percentage of missing data considered acceptable
    Returns: None
    '''
    prozent = df.isnull().mean()
    dropcol = list()
    for i in range(len(prozent)):
        if prozent[i] >= cutoff:
            dropcol.append(prozent.index.values[i])
    return dropcol

def encode_nominal(df, col):
    '''Convert labels to integers in the entries of a given column from a given dataframe.
    A simple encoding ensuring that the missing data is preserved for later imputation.
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
             col (String): Name of a nominal categorical feature to encode
    Returns: None
    '''
    print('\033[93m' + "[{:<30}](Unencoded) ".format(col), end=": ")
    print(df[col].unique())
    df[col] = df[col].astype("category")
    mapping = dict(enumerate(df[col].cat.categories))
    df[col] = df[col].cat.codes
    df.loc[(df[col] == -1 ), col] = np.nan
    print('\033[92m' + "{:<3}(  Encoded) ".format(" "), end=": ")
    print(df[col].unique())
    print('\033[94m' + "Mapping ==>> {}".format(mapping))

def encode_ordinal(df, col, catchall):
    '''Disambiguate phantom duplicate categories and replace catch-all with next bigger
    integer in the feature to preserve the order information.
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
             col (String): Name of a nominal categorical feature to encode
             catchall (String): Default bin for unknwon, not the same as NaN
    Returns: None
    '''
    print('\033[93m' + "[{:<30}](Unencoded) ".format(col), end=": ")
    print(df[col].unique())

    df[col] = df[col].fillna(-1) # Hide NaNs as siloed int
    for i in range(1,10):
        df.loc[(df[col] == str(i) ), col] = int(i)
        df.loc[(df[col] == i ), col] = int(i)
    df.loc[(df[col] == catchall), col] = int(10) # Encode catch-all bin as next in pecking order
    df[col] = df[col].astype("int")
    df.loc[(df[col] == -1 ), col] = np.nan
    print('\033[92m' + "{:<32}(  Encoded) ".format(" "), end=": ")
    print(df[col].unique())

def impute_feature(col):
    '''Carry out the imputation technique on one caegorical feature.
    Args:    col ((N,) shaped structure): Any feature from General Population, Customers, or Mailout datasets
    Returns: None
    '''
    bin_totals = col.value_counts()
    bin_values = bin_totals.index
    frequencies = [x for x in bin_totals.values/bin_totals.values.sum()]
    col.fillna(col.isnull()*np.random.choice(bin_values, len(col), p=frequencies), inplace=True)

def impute(df, min_bins=3, max_bins=50, max_miss=15):
    '''Carry out the automated imputation of categorical features in a dataframe.
       This function is merely a sanity filter to the impute_feature(col) routine above.
       The EXCEL spreadsheet 'DIAS Attributes - Values 2017.xls' shows categorical features
       having typically 10 or less slots, with some exceptions, e.g., 'LP_LEBENSPHASE_FEIN'
       which has 40. We use 50 as a threshold to decide if a feature is categorical.
       Numerical features in a dataset nearing a million rows are bound to have significantly
       more than 100 differing entries. We also set an arbitrary limit (15%) as the maximum
       fraction of missing data that we are willing to trust this automated methodology to impute.
       Beyond this level, we intend to handle columns manually, i.e., with visualization.
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
             min_bins(integer, default 2): Minimum number of slots in the feature to use for imputing
             max_bins(integer, default 50): Beyond this number of bins we must be dealing with numerical data
             max_miss(float, default 15): Max %-age of missing data to impute with this methodology
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
                    print('\033[91m' + "{:<28} Found {:>2} slots, NOT IMPUTING {:>2}%".format(feat, bins_count, isnan_pcent))
                else:
                    impute_feature(df[feat])
                    print('\033[92m' + "{:<28} Found {:>2} slots, imputing {:>2}% with {} and p={}"\
                        .format(feat, bins_count, isnan_pcent, bin_values.tolist(), [round(foo,2) for foo in frequencies]))


def see_impute(df):
    '''Show columns of a dataframe (with NaNs as bin value -10) next to the results of imputation
    Args:    df (DataFrame): A subset of either General Population, Customers, or Mailout datasets
    Returns: None
    '''
    NAN_VALUE = -1
    for feat in df:
        withnan = copy.deepcopy(df[feat])
        wcolors = ['gray' for x in withnan.value_counts(dropna=False).sort_index().values]
        wcolors[0] = 'red'
        withnan.fillna(value=NAN_VALUE, inplace=True)
        imputed = copy.deepcopy(df[feat])
        imputed.fillna(imputed.isnull()*np.random.choice(imputed.value_counts().index, len(imputed), \
                p=[x for x in imputed.value_counts().values/imputed.value_counts().values.sum()]), inplace=True)
        icolors = ['black' for x in imputed.value_counts().values]
        f = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT/2))
        ax1 = f.add_subplot(121)
        ax1.set_title(feat + " ---- BEFORE IMPUTATION")
        plt.bar(withnan.value_counts().sort_index().index, withnan.value_counts().sort_index().values, color=wcolors)
        ax2 = f.add_subplot(122, sharex=ax1, sharey=ax1)
        ax2.set_title(feat + " ---- AFTER IMPUTATION")
        plt.bar(imputed.value_counts().index, imputed.value_counts().values, color=icolors)
        plt.show()

def mean_or_not(col, xrange):
    '''Show the data, the result of mean imputation, and the results of our imputation on some integer
       numerical features of interest, particulary KBA13_ANZAHL_PKW.
    Args: col ((N,) shaped structure): Any numerical feature from General Population, Customers, or Mailout datasets
          slots (Integer): Number of bins to use for the histogram.
    Returns: None
    '''
    NAN_VALUE = -1
    withnan = copy.deepcopy(col)
    withnan.fillna(value=NAN_VALUE, inplace=True)
    meaned = copy.deepcopy(col)
    meaned.fillna(col.mean(), inplace=True)
    imputed = copy.deepcopy(col)
    imputed.fillna(imputed.isnull()*np.random.choice(imputed.value_counts().index, len(imputed), \
            p=[x for x in imputed.value_counts().values/imputed.value_counts().values.sum()]), inplace=True)

    f = plt.figure(figsize=(1.2*FIG_WIDTH, FIG_HEIGHT))
    ax3 = f.add_subplot(133)
    _ = sns.distplot(imputed, hist=False, kde_kws={'clip': xrange})
    _ = ax3.set_title("IMPUTED", fontsize=15)
    _ = ax3.set_xlabel(col.name, fontsize=15)
    _ = ax3.set_ylabel('Density', fontsize=15)
    ax1 = f.add_subplot(131, sharex=ax3, sharey=ax3)
    _ = sns.distplot(withnan, hist=False, kde_kws={'clip': xrange})
    _ = ax1.set_title("RAW DATA", fontsize=15)
    _ = ax1.set_xlabel(col.name, fontsize=15)
    _ = ax1.set_ylabel('Density', fontsize=15)
    ax2 = f.add_subplot(132, sharex=ax3, sharey=ax3)
    _ = sns.distplot(meaned, hist=False, kde_kws={'clip': xrange})
    _ = ax2.set_title("MEANED", fontsize=15)
    _ = ax2.set_xlabel(col.name, fontsize=15)
    _ = ax2.set_ylabel('Density', fontsize=15)
    _ = plt.show()

def numericals(df):
    '''List numerical features to check for possible outliers
    Args:    df (DataFrame): General Population, Customers, or Mailout datasets
    Returns: None
    '''
    MAX_CATEGORIES = 50 # Arbitrary cutoff to decide if a feature has numerical characteristics (too many slots)
    for feat in df:
        bin_counts = df[feat].value_counts().count()
        if bin_counts > MAX_CATEGORIES:
            print("{:<30} [{:>4}]".format(feat, bin_counts))
