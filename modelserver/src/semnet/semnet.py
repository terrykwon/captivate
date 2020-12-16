import heapq # prioirty queue
from dataclasses import dataclass, field

import numpy as np
from konlpy.tag import Twitter
from konlpy.tag import Kkma
from soylemma import Lemmatizer
from soynlp.noun import LRNounExtractor_v2

''' Things to consider:
    - Does the semantic network itself have state? Or is it just a static graph?
        -> for now just static embeddings

'''

twitter = Twitter()
tagger = Kkma()
lemmatizer = Lemmatizer()

import io

def load_vectors(fname):
    ''' Method for loading vectors from fasttext pretrained embeddings.
    '''
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def parse_utterance(u):
    ''' Parses the utterance and yields
        1) nouns
        2) lemmatized verbs and adjectives
    '''
    results = []
    morphemes = tagger.pos(u)

    nouns = tagger.nouns(u)
    verbs = [m[0] for m in morphemes if 'VV' in m[1]]
    # other = ? TODO

    results.extend(nouns)
    results.extend([lemmatizer.lemmatize(v) for v in verbs])
    results.extend([lemmatizer.lemmatize(w) for w in other])

    return results

def normalize(array):
    norm = np.linalg.norm(array)
    return array / norm

def create_word_vector(word):
    pos_list = twitter.pos(word, norm=True)
    word_vector = np.sum([pos_vectors.word_vec(str(pos).replace(" ", "")) for pos in pos_list], axis=0)
    return normalize(word_vector)


def get_pruned_vectors(wordlist, embeddings):
    ''' Saves a new file of keyed embeddings corresponding to only
        the words in the basic word list.
    '''
    with open(wordlist, 'r') as f:
        raw = f.read()
        basic_words = raw.split()
    
    with open(embeddings, 'w') as f:
        for word in basic_words:
            try:
                vec = create_word_vector(word)
                f.write('{} {}\n'.format(word, ' '.join(vec.astype('str'))))
            except KeyError:
                print('word not found: {}'.format(word))



def lemmatize(word):
    ''' The word should be converted to "base" form:
        e.g.) "뛰어갔어" -> "뛰다"
    '''
    pass


@dataclass(order=True)
class Candidate:
    ''' Just a weighted word data structure that can be used in a priority queue.
    '''
    relevance: float
    word: str = field(compare=False)


def initialize(word):
    ''' Returns a list of target words, weighted by importance.
        The weight is likely to be cosine distance.
        
        The weights don't need to be normalized since only the relative order
        of words matter.
    '''
    candidates = []

    for i in range(10):
        # Since heapq creates a min-heap, all the importance values should be negated
        candidate = Candidate(-i, "{}{}".format(word, i))
        heapq.heappush(candidates, candidate)

    return candidates


class SemanticNetwork:
    ''' 
        SPECS:
          - Given a query word, the network should return a graph of related words.

          - Additionally, each word should have an associated count of how many times
            it has been spoken.

          - "Difficult" words should be excluded as this is basically a word recommender
            for infants.


        Option 1:
          - Prune existing lexical networks like WordNet (English), KorLex (Kor)

        Option 2:
          - Create own network somehow
          - Using children's word learning data? 
              - But this would be enough to qualify as separate research.

        Option 3:
          - Ditch intervention altogether and focus on monitoring + analytics.
          - Would need more accurate interest detection model in this case.
   '''

    def __init__(self):
        pass