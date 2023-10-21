from googlesearch import search
import requests
from bs4 import BeautifulSoup
import justext
import re


# scrap results_urls
def get_html(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Check for any errors in the HTTP request
        html_content = response.text  # Get the HTML content
        return html_content
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve HTML: {e}")
        return ""


def remove_html_tags(input):
    soup = BeautifulSoup(input, "html.parser")
    text = soup.find_all(text=True)

    output = ""
    blacklist = [
        "[document]",
        "noscript",
        "header",
        "html",
        "meta",
        "head",
        "input",
        "script",
        "footer",
        "style",
        # there may be more elements you don't want, such as "style", etc.
    ]

    for t in text:
        if t.parent.name not in blacklist:
            output += "{} ".format(t)

    return output


def condense_newline(text: str) -> str:
    """Condenses multiple newlines into one newline"""
    return "\n".join([p for p in re.split("\n|\r", text) if len(p) > 0])


def test_removal(html_content: str):
    try:
        paragraphs = justext.justext(html_content, justext.get_stoplist("Polish"))
        new_paragraphs = []
        for paragraph in paragraphs:
            if not paragraph.is_boilerplate:
                new_paragraphs.append(paragraph.text)
        return condense_newline("\n".join(new_paragraphs))
    except Exception as e:
        print(e)
        return ""


def remove_paragraph_containing(text: str, substring: list) -> str:
    """Removes paragraphs containing any of the substrings"""
    paragraphs = text.split("\n")
    new_paragraphs = []
    for paragraph in paragraphs:
        if not any(sub in paragraph for sub in substring):
            new_paragraphs.append(paragraph)
    return "\n".join(new_paragraphs)


def get_data_for_llm(query):
    results_urls = [
        result
        for result in search(query, num=20, stop=20, pause=2)
        if "pl" in result
        and not any(
            domain in result for domain in ["onet", "facebook", "twitter", "pdf"]
        )
    ]

    html_documents = [get_html(url) for url in results_urls]
    # clean_html_documents = [remove_html_tags(html).strip() for html in html_documents]
    clean_html_documents = [
        f"<DOKUMENT_{i+1}>{test_removal(html)}</DOKUMENT_{i+1}>"
        for i, html in enumerate(html_documents)
        if html != ""
    ]
    ## sort by length
    clean_html_documents = sorted(clean_html_documents, key=len, reverse=False)
    # remove documents shorter than 100 characters
    clean_html_documents = [doc for doc in clean_html_documents if len(doc) > 100]
    ## print length of each document
    for doc in clean_html_documents:
        print(len(doc))
    clean_html_documents = "".join(clean_html_documents)
    # lower case
    clean_html_documents = clean_html_documents.lower()
    word_blacklist = [
        "komentarze",
        "komentarz",
        "reklama",
        "zaloguj",
        "zarejestruj",
        "cookies",
    ]
    clean_html_documents = remove_paragraph_containing(
        clean_html_documents, word_blacklist
    )

    return clean_html_documents[:12000]


def construct_prompt_to_use_source(USER_TAG, ASSISTANT_TAG, query):
    source = get_data_for_llm(query)
    prompt = f"{USER_TAG} Odpowiedz krótko i zwięźle na następujące pytanie: {query} korzystając tylko z tego tekstu źródłowego: {source}. {ASSISTANT_TAG}"
    return prompt
