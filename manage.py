from utils import preprocess_data
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

foldername = 'data'

logger.info('Started to unzip raw data.')
try:
    preprocess_data.unzip_file(foldername)
except FileNotFoundError:
    logger.warning('No raw files exists.')

logger.info('Started to process data.')
try:
    df_product_category, df_rating = preprocess_data.process_data(foldername)
except FileNotFoundError:
    logger.warning('No data exists.')