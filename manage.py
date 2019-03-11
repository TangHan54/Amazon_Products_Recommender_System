from utils import preprocess_data
import logging

logger = logging.getLogger(__name__)

foldername = 'data'

logger.warn('Started to unzip raw data.')
try:
    preprocess_data.unzip_file(foldername)
except FileNotFoundError:
    logger.warn('No raw files exists.')

logger.warn('Started to process data.')
try:
    df_product_category, df_rating = preprocess_data.process_data(foldername)
except FileNotFoundError:
    logger.warn('No data exists.')
    
print('df_product_category has ' + str(df_product_category.count()) +' rows.')
print('df_rating has ' + str(df_rating.count()) +' rows.')

logger.warn('Data is ready.')

