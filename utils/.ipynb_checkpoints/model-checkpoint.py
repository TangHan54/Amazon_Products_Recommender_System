import pickle
import logging
import os
from utils import preprocess_data
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, IndexToString, StringIndexerModel
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import array, col, lit, struct
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)
fpath = os.path.abspath('')
spark = SparkSession \
        .builder \
        .appName("Amazon Recommender System") \
        .config("spark.driver.maxResultSize", "64g") \
        .config("spark.driver.memory", "64g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.master", "local[10]") \
        .getOrCreate()


def train(foldername='data'):
    """
        input_id
        foldername: the folder that contains the data.
        recommend_for: 'product' to recommend a list of users for a product
            or 'user' to recommend a list of product for a user.
        number_of_recommendations: default 10.
    """
    # Preprocess data
    logger.warning('Started to unzip raw data.')
    try:
        preprocess_data.unzip_file(foldername)
    except FileNotFoundError:
        logger.warning('No raw files exists.')

    logger.warning('Started to process data.')
    try:
        df_product_category, df_rating = preprocess_data.process_data(spark,foldername)
    except FileNotFoundError:
        logger.warning('No data exists.')


    # create index for string type columns
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(df_rating) for col in df_rating.columns if col != 'overall']
    pipeline = Pipeline(stages=indexers)
    df_rating= pipeline.fit(df_rating).transform(df_rating)
    [indexers[i].write().overwrite().save(fpath+'/index/'+str(i)) for i in range(len(indexers))]
    

    # split into train and test 
    df = df_rating.select(['reviewerID_index','productID_index','overall'])
    (train, test) = df.randomSplit([0.8, 0.2], seed=0)

    als = ALS(userCol="reviewerID_index", itemCol="productID_index", ratingCol="overall", coldStartStrategy="drop")
    # Add hyperparameters and their respective values to param_grid
    # Cross validation to 
    param_grid = ParamGridBuilder() \
                .addGrid(als.rank, [10, 20]) \
                .addGrid(als.maxIter, [5, 10]) \
                .addGrid(als.regParam, [.01, .05, .1, .15]) \
                .build()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="overall", predictionCol="prediction") 
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
    # train the model for testing
    model = cv.fit(train)
    model_path = fpath + "/model_test"
    model = model.bestModel
    model.write().overwrite().save(model_path)
    predictions = model.transform(test)
    rmse = evaluator.evaluate(predictions)
    print('The test RMSE is', round(rmse, 4))
    with open('test_rmse.pickle','wb') as f:
        pickle.dump(rmse, f)
    
    
    # train the model with complete dataset
    #     model = cv.fit(df)
    #     model_path = fpath + "/model_prod"
    #     model = model.bestModel
    #     model.write().overwrite().save(model_path)

def recommend(input_id='AXBNEFRD90GLM',recommend_for='user', number_of_recommendations=10):
    indexers = [StringIndexerModel.load(fpath+'/index/'+str(i)) for i in range(2)]
    model = ALSModel.load(fpath+'/model_test')
    
    n = number_of_recommendations
    if recommend_for == 'user':
    # Generate top 10 product recommendations for each user
        userRecs = model.recommendForAllUsers(n)
        converter = IndexToString(inputCol="reviewerID_index", outputCol="reviewerID", labels=indexers[0].labels)
        user_rec = converter.transform(userRecs)

        product_labels_ = array(*[lit(x) for x in indexers[1].labels])
        recommendations = array(*[
            struct
                (product_labels_[col("recommendations")[i]['productID_index']].alias("productId"),
                col("recommendations")[i]["rating"].alias("rating")) 
                for i in range(n)
            ]
        )
        user_rec = user_rec.withColumn("recommendations", recommendations).select(['reviewerID','recommendations'])
        user_rec.rdd.saveAsPickleFile(fpath+'/result/user_rec')
        print('Recommendation Successful!')
        print(user_rec.show(5))
        # user_rec.rdd.write().overwrite().saveAsPickleFile(fpath+'/recommendation/user_rec.pickle')
        if input_id:
            result = user_rec.where(user_rec.reviewerID == input_id)\
                .select("recommendations.productId", "recommendations.rating").collect()
            with open(fpath + f'result/{input_id}.pickle') as f:
                f.dump(result,f)
    else:
        # Generate top 10 user recommendations for each product
        productRecs = model.recommendForAllItems(10)
        converter = IndexToString(inputCol="productID_index", outputCol="productID", labels=indexers[1].labels)
        product_rec = converter.transform(productRecs)

        user_labels_ = array(*[lit(x) for x in indexers[0].labels])
        recommendations = array(*[
            struct
                (user_labels_[col("recommendations")[i]['reviewerID_index']].alias("userId"),
                col("recommendations")[i]["rating"].alias("rating")) 
                for i in range(n)
            ]
        )
        product_rec = product_rec.withColumn("recommendations", recommendations)
        print('Recommendation Successful!')
        print(product_rec.show(5))
        # product_rec.rdd.saveAsPickleFile(fpath+'/recommendation/product_rec.pickle')
        if input_id:
            return product_rec.where(product_rec.productID_index == input_id)\
                .select("recommendations.userId", "recommendations.rating").collect()