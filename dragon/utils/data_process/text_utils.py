from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
import re
import torch


#: Control characters.
CONTROLS = {
    '\u0001', '\u0002', '\u0003', '\u0004', '\u0005', '\u0006', '\u0007', '\u0008', '\u000e', '\u000f', '\u0011',
    '\u0012', '\u0013', '\u0014', '\u0015', '\u0016', '\u0017', '\u0018', '\u0019', '\u001a', '\u001b',
}
# There are further control characters, but they are instead replaced with a space by unicode normalization
# '\u0009', '\u000a', '\u000b', '\u000c', '\u000d', '\u001c',  '\u001d', '\u001e', '\u001f'


#: Hyphen and dash characters.
HYPHENS = {
    '-',  # \u002d Hyphen-minus
    '‐',  # \u2010 Hyphen
    '‑',  # \u2011 Non-breaking hyphen
    '⁃',  # \u2043 Hyphen bullet
    '‒',  # \u2012 figure dash
    '–',  # \u2013 en dash
    '—',  # \u2014 em dash
    '―',  # \u2015 horizontal bar
}

#: Minus characters.
MINUSES = {
    '-',  # \u002d Hyphen-minus
    '−',  # \u2212 Minus
    '－',  # \uff0d Full-width Hyphen-minus
    '⁻',  # \u207b Superscript minus
}

#: Plus characters.
PLUSES = {
    '+',  # \u002b Plus
    '＋',  # \uff0b Full-width Plus
    '⁺',  # \u207a Superscript plus
}

#: Slash characters.
SLASHES = {
    '/',  # \u002f Solidus
    '⁄',  # \u2044 Fraction slash
    '∕',  # \u2215 Division slash
}

#: Tilde characters.
TILDES = {
    '~',  # \u007e Tilde
    '˜',  # \u02dc Small tilde
    '⁓',  # \u2053 Swung dash
    '∼',  # \u223c Tilde operator #in mbert vocab
    '∽',  # \u223d Reversed tilde
    '∿',  # \u223f Sine wave
    '〜',  # \u301c Wave dash #in mbert vocab
    '～',  # \uff5e Full-width tilde #in mbert vocab
}

#: Apostrophe characters.
APOSTROPHES = {
    "'",  # \u0027
    '’',  # \u2019
    '՚',  # \u055a
    'Ꞌ',  # \ua78b
    'ꞌ',  # \ua78c
    '＇',  # \uff07
}

#: Single quote characters.
SINGLE_QUOTES = {
    "'",  # \u0027
    '‘',  # \u2018
    '’',  # \u2019
    '‚',  # \u201a
    '‛',  # \u201b

}

#: Double quote characters.
DOUBLE_QUOTES = {
    '"',  # \u0022
    '“',  # \u201c
    '”',  # \u201d
    '„',  # \u201e
    '‟',  # \u201f
}

#: Accent characters.
ACCENTS = {
    '`',  # \u0060
    '´',  # \u00b4
}

#: Prime characters.
PRIMES = {
    '′',  # \u2032
    '″',  # \u2033
    '‴',  # \u2034
    '‵',  # \u2035
    '‶',  # \u2036
    '‷',  # \u2037
    '⁗',  # \u2057
}

#: Quote characters, including apostrophes, single quotes, double quotes, accents and primes.
QUOTES = APOSTROPHES | SINGLE_QUOTES | DOUBLE_QUOTES | ACCENTS | PRIMES

def normalize(text):
    for control in CONTROLS:
        text = text.replace(control, '')
    text = text.replace('\u000b', ' ').replace('\u000c', ' ').replace(u'\u0085', ' ')

    for hyphen in HYPHENS | MINUSES:
        text = text.replace(hyphen, '-')
    text = text.replace('\u00ad', '')

    for double_quote in DOUBLE_QUOTES:
        text = text.replace(double_quote, '"')  # \u0022
    for single_quote in (SINGLE_QUOTES | APOSTROPHES | ACCENTS):
        text = text.replace(single_quote, "'")  # \u0027
    text = text.replace('′', "'")     # \u2032 prime
    text = text.replace('‵', "'")     # \u2035 reversed prime
    text = text.replace('″', "''")    # \u2033 double prime
    text = text.replace('‶', "''")    # \u2036 reversed double prime
    text = text.replace('‴', "'''")   # \u2034 triple prime
    text = text.replace('‷', "'''")   # \u2037 reversed triple prime
    text = text.replace('⁗', "''''")  # \u2057 quadruple prime
    text = text.replace('…', '...')   # \u2026
    text = text.replace(' . . . ', ' ... ')  

    for slash in SLASHES:
        text = text.replace(slash, '/')

    for tilde in TILDES:
        text = text.replace(tilde, '~')

    return text


def embed_texts(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        texts: list[str], 
        batch_size: int, 
        text_size: int, 
        text_loader: callable,
        **kwargs
    ):
    allids, allembeddings = [], []
    # total = (len(texts) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_ids, batch_text in text_loader(texts, batch_size, **kwargs):
            encoded_batch = tokenizer.batch_encode_plus(
                batch_text, return_tensors="pt",
                max_length=text_size,
                padding=True, truncation=True
            )
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
            embeddings = model(**encoded_batch)

            allids.extend(batch_ids)
            allembeddings.append(embeddings)

    allembeddings = torch.cat(allembeddings, dim=0).cpu().numpy()
    return allids, allembeddings


def remove_broken_sentences(
        passage: str, from_beg=True, from_end=True) -> str:
    """
    Removes the incomplete sentences at the beginning and end of a passage.
    
    @param passage: The passage to process.
    @param from_beg: Whether to remove incomplete sentences from the beginning.
    @param from_end: Whether to remove incomplete sentences from the end.
    
    @return: The passage with incomplete sentences removed.
    """
    if from_beg:
        # Remove any text before the first sentence-ending punctuation
        passage = re.sub(r'^.*?[.!?]\s', '', passage, flags=re.DOTALL)

    if from_end:
        # Remove any text after the last sentence-ending punctuation
        passage = re.sub(r'([.!?])[^.!?]*$', r'\1', passage, flags=re.DOTALL)

    return passage


def ends_mid_sentence(passage):
    return not re.search(r'[.!?"]$', passage)


def get_incomplete_sentence(passage, from_end=False):
    if from_end:
        sentences = re.split(r'\s*[.!?]\s+', passage)
        return sentences[-1].rstrip()
    else:
        match = re.match(r'^.*?[.!?]', passage)
        if match:
            return match.group()
        else:
            print("Warning: cannot detect sentence boundary", passage)
            return passage