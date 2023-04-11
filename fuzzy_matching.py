import difflib
from nltk.util import ngrams
from fuzzywuzzy import process

def generate_ngrams(tokens, n):
  grams = list(ngrams(tokens, n))
  grams = [' '.join(item) for item in grams]
  return grams
 
def accumulate_ngrams(tokens, n):
  all_grams = []
  for i in range(1, n+1):
    all_grams += generate_ngrams(tokens, i)
  return all_grams
 
def fuzzy_matching(review, predict_token):
  tokens = review.split()
  n = len(predict_token.split())
  all_grams = accumulate_ngrams(tokens, n+1)
  print(all_grams)
  # return difflib.get_close_matches(predict_token, all_grams, 1, cutoff=0.1)
  return process.extractOne(predict_token, all_grams)[0]