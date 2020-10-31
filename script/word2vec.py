import os
import re
import spacy
from gensim.models import Word2Vec

def normalize(text):
    # convert to lowercase, 
    text = text.lower()
    
    # remove non-alphanumeric chars, numbers
    text = re.sub(pattern='[^a-zA-Z\s]+', repl='', string=text)
    
    # remove multiple spaces to single space
    text = re.sub(pattern='\s+', repl=' ', string=text)
    
    # strip the spaces in the beginning or end
    text = text.strip()
    
    return text

# tokenization & lemmatization
# spacy
nlp = spacy.load('en_core_web_sm') 
nlp.add_pipe(nlp.create_pipe('sentencizer'))

def tokenize(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    stem_tokens = [token.lemma_ for token in doc]
    stem_string = ' '.join(stem_tokens)
    return stem_string


if __name__ == "__main__":
    file = "sent_bert_raw.txt"
    clean_file = "clean_text.txt"

    # load the text
    with open(file, 'r') as f:
        sentences = f.readlines()
    
    # process the text
    tokenized_sentences = list(map(tokenize, sentences))
    normalized_sentences = list(map(normalize, tokenized_sentences))

    # output to a file
    with open(clean_file, 'w') as file:
    for sent in normalized_sentences:
        file.write(sent)
        file.write('\n')

    # split the corpus
    split_sentences = [sent.split() for sent in normalized_sentences]
    
    # train a series of models
    model = Word2Vec(split_sentences, size=100, window=5, min_count=1, workers=4)
    # change embeded size
    model_size5 = Word2Vec(split_sentences, size=5, window=5, min_count=1, workers=4)
    # CBOW model
    model_cbow = Word2Vec(split_sentences, size=100, window=5, min_count=1, workers=4, sg=0)
    # skip-gram model
    model_skipgram = Word2Vec(split_sentences, size=100, window=5, min_count=1, workers=4, sg=1)
    # larger window size
    model_window20 = Word2Vec(split_sentences, size=100, window=20, min_count=1, workers=4, sg=0)

    # evaluate the result
    for m in [model, model_size5, model_cbow, model_skipgram, model_window20]:
        print(m.similar_by_word("python",10))
        print(m.similar_by_word("neural",10))