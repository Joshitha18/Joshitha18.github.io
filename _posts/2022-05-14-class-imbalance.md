---
title: Handling Imbalanced data - part 1
tags: [machine learning]
style:  border
color: primary
comments: true
description: Imbalanced data refers to classification problems where we have unequal instances for different classes. In this post, we'll discuss the concepts in balancing data.
---


When faced with a class imbalance in our data, we may want to try to balance the training data before we build a model around it. One way to do this is by using sampling techniques which focus on solving the issue through manipulation of the data - we modify the data distribution to make sure that the data is balanced.


The two most adopted sampling methods are: 
1. Oversampling the minority class 
2. Undersampling the majority class

{% include elements/figure.html image="https://rb.gy/b6qg8i" caption="sampling methods" %}

## Oversampling the minority class 

In the case of over-sampling, we pick a larger proportion from the class with fewer values in order to come closer to the amount of the majority class; here we increase the number of data points by either randomly duplicating instances of the minority classes, or generating new data similar to the values in the existing data. 

We can use `RandomOverSampler` function from `imbalanced-learn` module to perform oversampling:

```python
import numpy as np
from imblearn.over_sampling import RandomOverSampler

data = np.random.randn(30,2)
label = 25*[0] + 5*[1]


# setting sampling_strategy=0.5 ensures that the minority class will be oversampled to have half the number of examples as the majority class

ros = RandomOverSampler(sampling_strategy=0.5, random_state=123)
data_res, label_res = ros.fit_resample(data, label)
```

Here, the over-sampling of the minority class is done by picking samples at random with replacement. It supports multi-class resampling by sampling each class independently and heterogeneous data as object array containing string and numeric data.


## Undersampling the majority class

Under-sampling, on the other hand, will take less data overall by reducing the amount taken from the majority class. So, it reduces the amount of data available to train our model - this means we should only use this if we have enough data that we can accept eliminating some of it. 

`RandomUnderSampler` function from `imbalanced-learn` module to perform random undersampling:

```python
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

data = np.random.randn(30,2)
label = 25*[0] + 5*[1]


# setting sampling_strategy=0.5 ensures that the minority class is 50 percent of the majority class

ros = RandomUnderSampler(sampling_strategy=0.5, random_state=123)
data_res, label_res = ros.fit_resample(data, label)
```

The decision to use over-sampling or under-sampling will depend on the amount of data we started with, and in some cases, computational costs. 


## Synthetic sampling(SMOTE)

It's clear that with smaller datasets, it won't be beneficial to under-sample. Instead, we can try over-sampling the minority class. Instead of randomly over-sampling with the `RandomOverSampler`, we can use the Synthetic Minority Over-sampling Technique (SMOTE) to create synthetic data.

*“This paper shows that a combination of our method of over-sampling the minority (abnormal) class and under-sampling the majority (normal) class can achieve better classifier performance (in ROC space) than only under-sampling the majority class. This paper also shows that a combination of our method of over-sampling the minority class and under-sampling the majority class can achieve better classifier performance (in ROC space) than varying the loss ratios in Ripper or class priors in Naive Bayes. Our method of over-sampling the minority class involves creating synthetic minority class examples.”*  -  **N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/pdf/1106.1813.pdf), Journal of Artificial Intelligence Research, 321-357, 2002** 

{% include elements/figure.html image="https://rb.gy/chn0oe" caption="SMOTE for synthetic data generation" %}

The main idea is to consider the relationships that exist between samples and create new synthetic points along the segments connecting a group of neighbors. SMOTE finds out ‘k’ nearest neighbors of a data point in the minority class. After the nearest data points have been identified, SMOTE then creates some synthetic data points on the lines joining the primary point and the neighbors so that these data points share the similar features/characteristics of the other minority data points.



References: 
1. Hands-On Data Analysis with Pandas, Stefanie Molin
2. [Oversampling to remove class imbalance using SMOTE](https://medium.com/@asheshdas.ds/oversampling-to-remove-class-imbalance-using-smote-94d5648e7d35)