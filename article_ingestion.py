import wikipediaapi
from wikipediaapi import WikipediaPage, Namespace
from typing import List, Generator, Iterable
import gensim
import nltk
from nltk.corpus import stopwords

IGNORE_KEYWORDS = (
    'fiction',
)


def get_pages_for_category(root_category_name: str, pages_to_fetch=250) -> List[WikipediaPage]:
    wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    category_page = wiki.page(f"Category:{root_category_name}")
    gen = _get_pages(category_page)
    pages = [next(gen) for _ in range(pages_to_fetch)]
    return pages


def _get_pages(root_page: WikipediaPage) -> Generator[WikipediaPage, None, None]:
    category_pages = []
    for page in root_page.categorymembers.values():
        if any(keyword in page.title.lower() for keyword in IGNORE_KEYWORDS):
            continue
        if page.namespace == Namespace.CATEGORY:
                category_pages.append(page)
        else:
            yield page

    for category_page in category_pages:
        for subcategory_page in _get_pages(category_page):
            yield subcategory_page


def write_corpus(pages: Iterable[WikipediaPage], file_name='corpus.txt'):
    with open(file_name, 'w') as file:
        file.writelines(_get_all_sentences(pages))


def _get_all_sentences(pages: Iterable[WikipediaPage]):
    for page in pages:
        for sentence in (gensim.utils.tokenize(sent) for sent in nltk.sent_tokenize(page.text)):
            yield f"{' '.join([s for s in sentence if s not in stopwords.words('english')])}\n"
