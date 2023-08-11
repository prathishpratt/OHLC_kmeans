import copy      #to make deep copies of DataFrames
import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mplfinance.original_flavor import candlestick_ohlc      #This is a change as mpl_finance is depricated
import matplotlib.dates as mdates
from matplotlib.dates import (
    DateFormatter, WeekdayLocator, DayLocator, MONDAY
)
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import yfinance as yf

import pandas_datareader.data as web   # Refer to this video on how to use   PDR https://www.youtube.com/watch?v=t_vZDyQDUkk


def get_open_normalised_prices(symbol,start,end):
    """
    Obtains a pandas DataFrame containing open normalised prices
    for high, low and close for a particular equities symbol
    from Yahoo Finance. That is, it creates High/Open, Low/Open 
    and Close/Open columns.
    """

    yf.pdr_override() #web.DataReader() wasnt working so I installed Yfinance
    #https://stackoverflow.com/questions/74912452/typeerror-string-indices-must-be-integer-pandas-datareader
    
    df = web.get_data_yahoo(symbol, start, end) #format is web.DataReader('GE', 'yahoo', start='2019-09-10', end='2019-09-10')

    #Normalize based on Open
    df["H/O"] = df["High"]/df["Open"]
    df["L/O"] = df["Low"]/df["Open"]
    df["C/O"] = df["Close"]/df["Open"]

    #Drop other than these 3 columns
    df.drop(
        [
            "Open", "High", "Low", 
            "Close", "Volume", "Adj Close"
        ], 
        axis=1, inplace=True
    )
    
    return df

def plot_candlesticks(data, since):
    """
    Plot a candlestick chart of the prices,
    appropriately formatted for dates
    """
    df = copy.deepcopy(data)       #Take a deep copy of the data
    df = df[df.index >= since]     #Subset only from the 'since' date

    #When we reset the index, the old index is added as a column, and a new sequential index is used. 
    #That is from index as dates, it will make dates as a new col and keeps index as 0,1,2....
    df.reset_index(inplace=True)


    #Convert datetime objects to Matplotlib dates. 
    #It basically calculates the number of days from a set epoch, here the default is (default: '1970-01-01T00:00:00').
    #So '1970-01-02' will have value 1 and  '1970-01-10' will have value 9. 
    #So '2013-01-02' has a value of 15707.0
    df['date_fmt'] = df['Date'].apply(
        lambda date: mdates.date2num(date.to_pydatetime())       
    )


    # Visualization stuff
    #This code snippet configures the x-axis of a Matplotlib plot to display date values with major tick marks on Mondays
    #and minor tick marks on every day. It also formats the major tick labels to show abbreviated month names and day numbers.
    # Set the axis formatting correctly for dates
    # with Mondays highlighted as a "major" tick
    mondays = WeekdayLocator(MONDAY)
    alldays = DayLocator()
    weekFormatter = DateFormatter('%b %d')
    fig, ax = plt.subplots(figsize=(16,4))
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)

    # Plot the candlestick OHLC chart using black for
    # up days and red for down days
    csticks = candlestick_ohlc(
        ax, df[
            ['date_fmt', 'Open', 'High', 'Low', 'Close']
        ].values, width=0.6, 
        colorup='#000000', colordown='#ff0000'
    )
    ax.set_facecolor((1,1,0.9))                    #Changed set_axis_bgcolor with set_facecolor
    ax.xaxis_date()
    plt.setp(
        plt.gca().get_xticklabels(), 
        rotation=45, horizontalalignment='right'
    )
    plt.show()


def plot_3d_normalised_candles(data):
    """
    Plot a 3D scatterchart of the open-normalised bars
    highlighting the separate clusters by colour
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d',elev=21, azim=-135)   #If you put  elev=90, azim=-90, you can see it in 2d
    #This works ==> you do this in order to alert Matplotlib that we're using 3d data.
    # ax = Axes3D(fig, elev=21, azim=-136)   #This doesn't work
    np.float = float
    ax.scatter(
        data["H/O"], data["L/O"], data["C/O"], 
        c=labels.astype(np.float)
    )
    #ax.set_xlim([1, 1.04])
    ax.set_xlabel('High/Open')
    ax.set_ylabel('Low/Open')
    ax.set_zlabel('Close/Open')
    plt.show()

def plot_cluster_ordered_candles(data):
    """
    Plot a candlestick chart ordered by cluster membership
    with the dotted blue line representing each cluster
    boundary.
    """
    # Set the format for the axis to account for dates
    # correctly, particularly Monday as a major tick
    mondays = WeekdayLocator(MONDAY)
    alldays = DayLocator()
    weekFormatter = DateFormatter("")
    fig, ax = plt.subplots(figsize=(16,4))
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)


    # Sort the data by the cluster values and obtain
    # a separate DataFrame listing the index values at
    # which the cluster boundaries change
    df = copy.deepcopy(data)
    df.sort_values(by="Cluster", inplace=True)
    df.reset_index(inplace=True)
    df["clust_index"] = df.index
    df["clust_change"] = df["Cluster"].diff()
    change_indices = df[df["clust_change"] != 0]             #This change indices is the blue colour dotted lines in the plot

    # Plot the OHLC chart with cluster-ordered "candles"
    csticks = candlestick_ohlc(
        ax, df[
            ["clust_index", 'Open', 'High', 'Low', 'Close']
        ].values, width=0.6, 
        colorup='#000000', colordown='#ff0000'
    )
    ax.set_facecolor((1,1,0.9)) #Changed from ax.set_axis_bgcolor((1,1,0.9))

    # Add each of the cluster boundaries as a blue dotted line
    for row in change_indices.iterrows():
        plt.axvline(
            row[1]["clust_index"], 
            linestyle="dashed", c="blue"
        )
    plt.xlim(0, len(df))
    plt.setp(
        plt.gca().get_xticklabels(), 
        rotation=45, horizontalalignment='right'
    )
    plt.show()

def create_follow_cluster_matrix(data):
    """
    Creates a k x k matrix, where k is the number of clusters
    that shows when cluster j follows cluster i.
    """
    
    data["ClusterTomorrow"] = data["Cluster"].shift(-1)
    #Refer this for .shift()             https://www.geeksforgeeks.org/python-pandas-dataframe-shift/

    data.dropna(inplace=True)        #So the NANs in the below most row will be dropped, ie the last row will be dropped
    data["ClusterTomorrow"] = data["ClusterTomorrow"].apply(int)   #Convert Floats to Integers 

    #The zip() function in Python is commonly used to combine two or more iterables, such as lists, tuples, or strings, 
    #into a single iterable where elements from corresponding positions are paired together.
    sp500["ClusterMatrix"] = list(zip(data["Cluster"], data["ClusterTomorrow"])) 

    cmvc = data["ClusterMatrix"].value_counts() #return a Series containing counts of unique values.
    clust_mat = np.zeros( (k, k) )              #Creates empty square matrix named clust_mat with dimensions k by k


    #The value counts will return an iterable in "cmvc". In that iterable, we iterate row by row. For ex,
    # cluster     count
    #   1           42
    #   3           23
    #   2           14

    #Now for each row, we take the count col, ie row[1], then we see the % share of that cluster in the data
    for row in cmvc.items():            #iteritems(): was deprecated
        clust_mat[row[0]] = row[1]*100.0/len(data)
        
    return clust_mat


#Main
if __name__ == "__main__":
    # Obtain S&P500 pricing data from Yahoo Finance
    start = datetime.datetime(2013, 1, 1)
    end = datetime.datetime(2015, 12, 31)
    symbol = "^GSPC"
    yf.pdr_override()
    #S&P 500 (^GSPC) SNP - SNP Real Time Price. Currency in USD
    sp500  = web.get_data_yahoo(symbol,start, end)

    # Plot last year of price "candles"
    plot_candlesticks(sp500, datetime.datetime(2015, 1, 1))

    # Get the normalised data
    sp500_norm = get_open_normalised_prices(symbol, start, end)

    # Carry out K-Means clustering with five clusters on the three-dimensional data H/O, L/O and C/O
    k =5
    km = KMeans(n_clusters=k, random_state=42) #Random state is just an identity that python gives to your algo
    #Watch this  https://www.youtube.com/watch?v=rArvRWyH5Sk
    km.fit(sp500_norm)
    #You fit with sp500_norm norm data, but put the resulting labels into sp500 df for readability
    
    #Labels of each point
    labels = km.labels_
    sp500["Cluster"] = labels
    
    # Plot the 3D normalised candles using H/O, L/O, C/O
    plot_3d_normalised_candles(sp500_norm)

    # Plot the full OHLC candles re-ordered 
    # into their respective clusters    
    plot_cluster_ordered_candles(sp500)

    # Create and output the cluster follow-on matrix
    cust_mat = create_follow_cluster_matrix(sp500)