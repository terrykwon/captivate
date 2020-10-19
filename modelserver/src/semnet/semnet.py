


def lemmatize(word):
    ''' The word should be converted to "base" form:
        e.g.) "뛰어갔어" -> "뛰다"
    '''
    pass


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