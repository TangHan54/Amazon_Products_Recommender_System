from pyspark import SparkConf
from pyspark.sql import SparkSession, types
from pyspark.sql.functions import lit, count
import json
import os 


# The raw data is in .gz format
# Step 1. unzip the file
def unzip_file(foldername):
	items = os.listdir(foldername)

	for name in items:
		if name.endswith(".gz"):
			name = name.rstrip('.gz')
			if name not in items:
				os.system(f'gunzip {foldername}/{name}.gz')

# Step 2. Extract the category from the fname
# Step 3. Create a Dataframe of the format as (productid, category)

def process_data(spark, foldername, user_threshold=5, product_threshold=5):
	items = os.listdir(foldername)

	M = 999
	for idx, elem in enumerate(items):
		if elem.endswith('.json'):
			category = elem.lstrip('reviews_').rstrip('_5.json')
			elem = foldername + '/' + elem
			# add category as a column in the dataframe
			if idx < M:
				df = spark.read.json(elem)
				df = df.withColumn('category', lit(category)).select('asin','reviewerID','overall','category')
				M = idx
			else:
				temp_df = spark.read.json(elem)
				temp_df = temp_df.withColumn('category', lit(category)).select('asin','reviewerID','overall','category')
				df = df.unionAll(temp_df)

	# drop duplicates
	df = df.dropDuplicates()
	# shape of the dataframe
	print((df.count(), len(df.columns)))

	# Remove users who reviewed less than user_threshold
	user_df = df.groupby(df.reviewerID).agg(count(df.reviewerID).alias('nb'))
	user_df = user_df.filter(f"nb > {user_threshold}").select('reviewerID')
	df = df.join(user_df,'reviewerID', 'inner')
	# Remove products whose reviews are less than product_threshold
	product_df = df.groupby(df.asin).agg(count(df.asin).alias('nb'))
	product_df = product_df.filter(f"nb > {product_threshold}").select('asin')
	df = df.join(product_df, 'asin', 'inner').withColumnRenamed('asin','productID')

	df_product_category = df.select(['productID','category']) 
	df_rating = df.select(['reviewerID','productID','overall'])
	return df_product_category, df_rating