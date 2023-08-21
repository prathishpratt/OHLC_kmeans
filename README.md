
## K-Means Clustering of stock OHLC Data

K-Means Clustering will be used on daily Open-High-Low-Close (OHLC) data, also known as bars or candles. Such analysis is interesting because it considers extra dimensions to daily data that are often ignored, in favor of adjusting closing prices.

Each of the High, Low, and Close dimensions will be normalized by the corresponding Open price. So the number of dimentions will be reduced from 4 to 3. By normalizing each candle in this manner, the dimensionality is reduced from four (Open, High, Low, Close) to three (High/Open, Low/Open, Close/Open).





The 3D normalised candles using H/O, L/O, C/O :-
![Figure_2](https://github.com/prathishpratt/OHLC_kmeans/assets/64516584/f0e4c149-454d-485b-a7ca-74e6f00e11ce)


The price of "candles" :-
![Figure_1](https://github.com/prathishpratt/OHLC_kmeans/assets/64516584/6d4499bf-2856-48c6-b8b1-63226bffbe71)


The clustered plot, the blue dotted lines indicate the boundary :-
![Figure_3](https://github.com/prathishpratt/OHLC_kmeans/assets/64516584/88f80d0b-910b-415c-8222-d5f5f6604121)


## Notes:
This project is quite different from normal ones where we take say 5 stocks of OHLC data, normalize it, and call k-means. And it will cluster all tech stocks and energy stocks etc.

Here, we take one stock(S&P 500) and then we cluster OHLC data of each date, so a few days can be in cluster 0 and another few in cluster 3, etc. Going further, we also find the probability/frequency of today's OHLC price going from one cluster to another cluster.

So, if we are able to recognize the commonalities of each cluster, say one cluster has days where the stock was very volatile and another day it was very calm(O,C,H,L are all very close), from the cluster follow-on matrix we can predict how possible the clam stock today might be more volatile the next day.

- This application can work on any stock/commodity provided Yahoo has the data for it.

- More involvement can be done to fix the value of k, here we have assumed the value of k = 5, but if we use the Elbow Curve Method, that is we can perform the clustering with many different k and then plot with the Within-cluster sum of squares(WSS), we can find the optimal k.

- Another problem is that all of this work is in-sample. Any future usage of this as a predictive tool implicitly assumes that the distribution of clusters would remain similar to the past. A more realistic implementation would consider some form of "rolling" or "online" clustering tool that would produce a follow-on matrix for each rolling window.

- We can also use evaluation metrics on the clustering multiple times, find the best one and set the random state as the following observation for more optimal and stable results.
