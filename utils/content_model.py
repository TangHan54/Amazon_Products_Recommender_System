import logging
import os
from utils import preprocess_data
from pyspark.sql.functions import concat_ws, collect_list
from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, HashingTF, IDF, StopWordsRemover,Word2Vec
from pyspark.ml import Pipeline, PipelineModel

logger = logging.getLogger(__name__)
fpath = os.path.abspath('')
spark = SparkSession \
        .builder \
        .appName("Amazon Recommender System") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.master", "local[4]") \
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
    # group the reviews together for the same product
    df_content = df_rating.groupBy("productID").agg(concat_ws(" ", collect_list("reviewText")).alias("reviewText"))
    # tokenize
    regexTokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'reviewText', outputCol = 'token')
    stopWordsRemover = StopWordsRemover(inputCol = 'token', outputCol = 'nostopwrd')
    word2Vec = Word2Vec(vectorSize = 100, minCount = 5, inputCol = 'nostopwrd', outputCol = 'word_vec', seed=123)
    pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover, word2Vec])
    # fit the model
    pipeline_mdl = pipeline.fit(df_content)
    # save the model
    df_content = pipeline_mdl.transform(df_content)
