from article_ingestion import get_pages_for_category, write_corpus
from fasttext_utils import get_model, get_similarity_matrix, pp_matrix
import random
from logging import getLogger, INFO, StreamHandler, basicConfig
import sys
import os
import xlsxwriter


random.seed(101)
basicConfig(
    level=INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[StreamHandler(sys.stdout)]
)
logger = getLogger('upz_zavrsni')


TRAIN_CORPUS = 'train_corpus.txt'
TEST_CORPUSES = [f"doc{i+1}.txt" for i in range(10)]
MODEL_PATH = 'model_fasttext.bin'
WORKBOOK_PATH = 'similarity_matrix.xlsx'


if not all([os.path.isfile(path) for path in ([TRAIN_CORPUS] + TEST_CORPUSES)]):
    logger.info(f"Setting up training pages...")
    pages_train = get_pages_for_category('Artificial intelligence', pages_to_fetch=1000)
    random.shuffle(pages_train)

    logger.info(f"Setting up testing pages...")
    pages_test = [pages_train.pop() for _ in range(len(TEST_CORPUSES)-3)]
    pages_test.extend(get_pages_for_category('Beekeeping', pages_to_fetch=3))

    logger.info(f"Writing train corpus...")
    write_corpus(pages_train, TRAIN_CORPUS)

    logger.info(f"Writing test corpuses...")
    for i in range(len(TEST_CORPUSES)):
        write_corpus([pages_test[i]], file_name=TEST_CORPUSES[i])

logger.info("Loading model...")
model = get_model(MODEL_PATH, TRAIN_CORPUS)

logger.info("Calculating similarity...")
similarity_matrix = get_similarity_matrix(model, TEST_CORPUSES)

logger.info("Writing results to workbook...")
workbook = xlsxwriter.Workbook(WORKBOOK_PATH)
worksheet = workbook.add_worksheet()
for row, data in enumerate(similarity_matrix):
    worksheet.write_row(row, 0, data)
workbook.close()

logger.info("Done!")

