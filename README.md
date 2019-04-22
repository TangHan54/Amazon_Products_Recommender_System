# Amazon_Products_Recommender_System
CS5344 Big Data Project.

Data Source: [Amazon product data](http://jmcauley.ucsd.edu/data/amazon/)

Data should be in the folder data. 
- Raw data should be in the form '.json.gz' or '.json'

1. Create a virtual environment
> mkvirtualenv spark \
> workon spark \
> pip install -r requirements.txt 

2. Process data

Data processing includes 

- combine multiple files
- remove duplicates
- remove users who posted less than n reviews, default 5.  
- remove products which received less than n reviews.

3. Build models

Content-based model

- Calculate the tfidf for reviewText
- Calculate the pair-wise similarity
- Recommend the top ones given a product based on similarity

Collaborative Filtering

- Matrix Factorization based on Explicit Rating
- Given a user/product, recommend the top rating ones.

> spark-submit manage.py --master "local[8]"

or
> python manage.py


