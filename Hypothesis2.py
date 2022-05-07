import pandas as pd
from functools import partial, reduce
import matplotlib.pyplot as matplot
from scipy import stats


def get_conflicts():
    conf_df = pd.read_csv("data/ucdp-prio-acd-211.csv")
    proc_conf_df = conf_df[['location', 'year', 'intensity_level', 'type_of_conflict']]
    proc_loc = proc_conf_df.apply(lambda x: get_single_location(x.location), axis=1)
    proc_conf_df['location'] = proc_loc
    proc_conf_df = proc_conf_df.rename(columns={'location': 'Entity', 'year': 'Year'})
    proc_conf_df = (proc_conf_df[proc_conf_df['intensity_level'] == 2]).set_index(['Entity', 'Year'])
    conflicts = pd.read_csv("data/DP_LIVE_30042022010903788.csv")
    conflicts = add_country_col(conflicts)
    conflicts = add_aggregate_year(conflicts)
    conf_dfs = [conflicts, proc_conf_df[['type_of_conflict']]]
    conf_merged = partial(pd.merge, left_index=True, right_index=True, how='outer')
    conf_merge_df = pd.DataFrame(reduce(conf_merged, conf_dfs))
    conf_m_df = conf_merge_df.reset_index()
    grouped_df = conf_m_df.groupby(['Year'])
    grouped_df.plot()


def get_single_location(row):
    """ Extracts a single location from multi valued cell
    :param row: the cell containing multi valued data
    >>> get_single_location('Argentina, Luisiana')
    'Argentina'
    >>> get_single_location('Mayanmar (Burma)')
    'Mayanmar'
    >>> get_single_location(23)
    Traceback (most recent call last):
    AttributeError: 'int' object has no attribute 'split'
    """
    str_row = row.split(",")[0]
    str_row = str_row.split("(")[0]
    return str_row.strip()


def add_aggregate_year(conflict_df):
    """ Extract year from the Time column and group by Year for calculating mean
    """
    conflict_df['Year'] = conflict_df["TIME"].str.slice(0, 4)
    conflict_df = conflict_df.groupby(['Entity', 'Year']).agg({'Value': ['mean']})
    conflict_df.columns = ['Value']
    return conflict_df


def add_country_col(conflicts_df):
    """ Add Country column after extracting Country name from the Country code
    """
    country_df = (pd.read_csv("data/CountryNamesCodesBasic.csv"))[['3char country code', 'Country Name (usual)']]
    country_df = country_df.rename(columns={'Country Name (usual)': 'Entity', '3char country code': 'location'})
    country_df = country_df.set_index(['location'])
    conflicts_df = conflicts_df.reset_index()
    conflicts_df = conflicts_df.rename(columns={'LOCATION': 'location'})
    conflicts_df = conflicts_df.set_index(['location'])
    confl_dfs = [conflicts_df, country_df[['Entity']]]
    confl_merge_df = merge_dataframes(confl_dfs, 'inner')
    return confl_merge_df


def merge_dataframes(to_merge_df, merge_type):
    """ A generic function to merge a stacked dataframe on index with the merge type mentioned
    :param to_merge_df: the stacked dataframe
    :param merge_type: The type of merge i.e. inner, outer, left, right"""
    merge_query = partial(pd.merge, left_index=True, right_index=True, how=merge_type)
    merged_dataframe = pd.DataFrame(reduce(merge_query, to_merge_df))
    merged_dataframe = merged_dataframe.reset_index()
    return merged_dataframe


def get_oilprice_dataset() -> pd.DataFrame:
    """ Reads the csv file for oil prices and extracts the needed columns for further processing"""
    oil_var_df = pd.read_csv("data/crude-oil-price.csv")
    oil_var_df['Year'] = oil_var_df['date'].str.slice(0, 4)
    oil_var_df['Year'] = oil_var_df['Year'].astype(int)
    oil_var_df = oil_var_df[['Year', 'price', 'percentChange', 'change']]
    return oil_var_df


def get_armedConflict_dataset() -> pd.DataFrame:
    """ Reads the csv file for armed conflicts and extracts the needed columns for further processing"""
    conf_var_df = pd.read_csv("data/ucdp-prio-acd-211.csv")
    proc_var_conf_df = conf_var_df[['location', 'year', 'intensity_level', 'type_of_conflict']]
    return proc_var_conf_df


def plot_oilprice_vs_conflicts_year(oil_grouped_df1, join_df1s):
    """ Plots the graph of oil price v/s conflicts.
    :param oil_grouped_df1: oil price dataset grouped by years
    :param join_df1s: oil and conflicts dataset joined on year"""
    matplot.plot(oil_grouped_df1.index, oil_grouped_df1['price'])
    matplot.title('Oil Prices vs Conflicts')
    matplot.xlabel('Year')
    matplot.ylabel('Price in USD/Bbl')
    matplot.scatter(join_df1s['Year'], join_df1s['price'])
    matplot.show()


def plot_oilpriceChange_vs_conflicts_year(oil_grouped_df2, join_df2s):
    """ Plots the graph of oil price change v/s conflicts.
    :param oil_grouped_df2: oil price dataset grouped by years
    :param join_df2s: oil and conflicts dataset joined on year"""
    matplot.plot(oil_grouped_df2.index, oil_grouped_df2['change'])
    matplot.title('Oil Price change vs Conflicts')
    matplot.xlabel('Year')
    matplot.ylabel('')
    matplot.scatter(join_df2s['Year'], join_df2s['change'])
    matplot.show()


def plot_oil_type_of_conflicts(proc_conf_df_var1):
    """ Plots the graph for different type of conflicts. Conflict ype 1 refers to war where >5000 population
    were affected. Type 2 refers to inter state conflicts affecting <5000 population
    :param proc_conf_df_var1: conflicts dataset grouped by years
    """
    fig, axis_data = matplot.subplots(figsize=(15, 7))
    # use unstack()
    proc_conf_df_var1.groupby(['Year', 'intensity_level']).count()['Entity'].unstack().plot(ax=axis_data)


def plot_2009_2010_data(proc_conf_dff1):
    """ Plots the graph of oil and/or conflicts for the years 2005-2010
    :param proc_conf_dff1: conflicts dataset grouped by years"""
    proc_conf_df3 = proc_conf_dff1
    oil_df1 = proc_conf_dff1.reset_index()
    merged_proc_conf_df = proc_conf_df3.merge(oil_df1, on='Year', how='left')
    merged_conf_0510 = merged_proc_conf_df[(merged_proc_conf_df['Year'] >= 2005) & (merged_proc_conf_df['Year'] < 2010)]
    print(merged_conf_0510)
    merged_conf_0510 = merged_conf_0510.reset_index()
    merged_conf_0510_count = merged_conf_0510.groupby(['Year']).agg({'Entity': 'count', 'price': 'mean'})
    print(merged_conf_0510_count)
    plot_linear_graphs_0510(merged_conf_0510_count)

    # Calculate Pearson model for correlation
    get_pearson_for_price_conflicts(merged_proc_conf_df)


def plot_oil_prices_over_years(proc_conf_dff, oil_grouped_dff):
    """ Plots the graph of oil prices over the years vs conflicts over the years
    :param proc_conf_dff: conflicts dataframe grouped by years
    :param oil_grouped_dff: oil price data grouped by years"""
    proc_conf_grouped_df1 = proc_conf_dff.groupby(['Year']).head(1)
    proc_conf_grouped_df1 = proc_conf_grouped_df1.set_index(['Year'])
    join_dfs = proc_conf_grouped_df1.join(oil_grouped_dff, on='Year', how='left', lsuffix='_proc', rsuffix='_oil')
    join_dfs = join_dfs.reset_index()
    matplot.bar(join_dfs['Year'], join_dfs['price'])
    matplot.title('Oil Prices')
    matplot.xlabel('Year')
    matplot.ylabel('Price in USD/Bbl')
    matplot.figure(figsize=(300, 300))
    # matplot.scatter(join_dfs['Year'], join_dfs['price'])
    matplot.show()


def plot_linear_graphs_0510(merged_conf_0510_count1):
    """ Plots the graph for the oil price for the largest price change duration i.e. 2005-2010. The price along with
    number of conflicts per country is used to check if the oil price and conflicts are linearly correlated
    :param merged_conf_0510_count1: The merged data where mean of oil prices column is merged to the conflicts
    dataset"""
    # plot oil linear
    matplot.plot(merged_conf_0510_count1.index, merged_conf_0510_count1['price'])
    matplot.title('Oil Prices from 2005-2010')
    matplot.xlabel('Year')
    matplot.ylabel('')
    matplot.show()
    # plot number of conflicts vs oil price per year for 2005-2010
    matplot.plot(merged_conf_0510_count1.index, merged_conf_0510_count1['price'])
    matplot.title('Oil Prices from 2005-2010 vs number of conflicts')
    matplot.xlabel('Year')
    matplot.ylabel('')
    matplot.plot(merged_conf_0510_count1.index, merged_conf_0510_count1['Entity'])
    matplot.show()


def get_pearson_for_price_conflicts(merged_proc_conf_dff1):
    """ Calculate the correlation between the oil price and number of Conflicts
    """
    merged_proc_conf_df_pearson = merged_proc_conf_dff1.groupby(['Year']).agg({'Entity': 'count', 'price': 'mean'})
    merged_proc_conf_df_pearson = merged_proc_conf_df_pearson.reset_index()
    merged_proc_conf_df_pearson_1983 = merged_proc_conf_df_pearson[(merged_proc_conf_df_pearson['Year'] >= 1983)]
    print(merged_proc_conf_df_pearson_1983)
    stats.pearsonr(merged_proc_conf_df_pearson_1983['Entity'], merged_proc_conf_df_pearson_1983['price'])


def analyse_oil_conflicts():
    """This function is used as a single function for all the analysis conducted in the hypothesis 2. This can be
    called by another file or jupyter notebook"""
    # process oil dataset
    oil_df = get_oilprice_dataset()
    oil_grouped_df = oil_df.groupby(['Year']).agg(['mean'])
    oil_grouped_df.columns = ['price', 'percentChange', 'change']

    # process conflicts
    proc_conf_df = get_armedConflict_dataset()
    proc_loc = proc_conf_df.apply(lambda x: get_single_location(x.location), axis=1)
    proc_conf_df['location'] = proc_loc
    proc_conf_df = proc_conf_df.rename(columns={'location': 'Entity', 'year': 'Year'})
    proc_conf_df = proc_conf_df.sort_values(by='Year')

    # joining data
    proc_conf_grouped_df1 = proc_conf_df.groupby(['Year']).head(1)
    proc_conf_grouped_df1 = proc_conf_grouped_df1.set_index(['Year'])
    join_dfs = proc_conf_grouped_df1.join(oil_grouped_df, on='Year', how='left', lsuffix='_proc', rsuffix='_oil')
    join_dfs = join_dfs.reset_index()
    # plot oil conflicts along with year, intensity level and locations
    plot_oil_type_of_conflicts(proc_conf_df)
    # plot oil prices along with conflicts year
    plot_oil_prices_over_years(oil_grouped_df, join_dfs)
    # plot oil price changes along with years
    plot_oilpriceChange_vs_conflicts_year(oil_grouped_df, join_dfs)

    # Since we see a maximum price change for the duration 2005-2010
    plot_2009_2010_data(proc_conf_df)


if __name__ == '__main__':
    analyse_oil_conflicts()
