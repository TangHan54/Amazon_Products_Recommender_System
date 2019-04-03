# Key points


## Matrix Factorization
[reference](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
- A class of collaborative filtering algorithms used in recommender systems.
- Latent Factors:
    - [rank of latent factors](https://towardsdatascience.com/paper-summary-matrix-factorization-techniques-for-recommender-systems-82d1a7ace74)
    - [How to decide rank of latent factors](https://www.quora.com/Recommendation-Systems-How-can-I-decide-the-dimension-of-latent-factor-for-users-items-in-case-matrix-factorization-based-collaborative-filters)
    
- Funk SVD:
    - as a rating prediction problem, therefore it uses **explicit**s numerical ratings as user-item interactions.
- SVD++:
    - exploit all available interactions both explicit (e.g. numerical ratings) and implicit (e.g. likes, purchases, skipped, bookmarked).
    - Cold-start problem.

## ALS
- A learning algorithm to minimize loss function for matrix factorization. (another is stochastic gradient descent)
- [ALS](https://www.quora.com/What-is-the-Alternating-Least-Squares-method-in-recommendation-systems-And-why-does-this-algorithm-work-intuition-behind-this)


## item-based CF vs user-based CF
[Reference](https://medium.com/@cfpinela/recommender-systems-user-based-and-item-based-collaborative-filtering-5d5f375a127f)

[Comparison](https://medium.com/@wwwbbb8510/comparison-of-user-based-and-item-based-collaborative-filtering-f58a1c8a3f1d)
>However, for UBCF, there is no offline calculation, so all of the computational cost is online, which turns out that the predictions are literally very slow even for middle-size datasets due to the heavy online computational cost.

- MF does not belong to any of them. They are separate methods to generate recommender systems.

## Explicit vs Implicit feedback
[Deployment in Pyspark](https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html)
[Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)

## Pyspark config

[reference](https://spark.apache.org/docs/latest/configuration.html)

The configuration is quite important to debug and allocate computing power.

- Only in YARN
    - spark.driver.cores
    - spark.driver.maxResultSize 
    - spark.driver.memory

- spark.executor.memory
- spark.master
- spark.serializer

> Driver memory are more useful when you run the application, In yarn-cluster mode, because the application master runs the driver. Here you are running your application in local mode driver-memory is not necessary. You can remove this configuration from you job.

## CountVectorizer vs HashingTF vs word2vec