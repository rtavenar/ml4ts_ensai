(sec:intro_align)=
# Alignment-based metrics

The definition of adequate metrics between objects to be compared is at the
core of many machine learning methods (_e.g._, nearest neighbors, kernel
machines, _etc._).
When complex objects (such as time series) are involved, such metrics have to
be carefully designed
in order to leverage on desired notions of similarity.

Let us illustrate our point with an example.

```{figure} ../fig/kmeans.svg
---
width: 100%
name: kmeans_euc
---
$k$-means clustering with Euclidean distance. Each subfigure
represents series from a given cluster and their centroid (in red).
```

The figure above is the result of a $k$-means clustering that uses Euclidean
distance as a base metric.
One issue with this metric is that it is not invariant to time shifts, while
the dataset at stake clearly holds such invariants.

When using a shift-invariant similarity measure (discussed in our
{ref}`sec:dtw` section) at the core of $k$-means, one gets:

```{figure} ../fig/kmeans_dtw.svg
---
width: 100%
name: kmeans_dtw
---
$k$-means clustering with Dynamic Time Warping. Each subfigure
represents series from a given cluster and their centroid (in red).
```

This part of the course tackles the definition of adequate similarity
measures for time series and their use at the core of machine learning methods.
