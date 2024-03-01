from time import time

ADJ_FEATS, PAST_VERB_FEATS = {"ADJF", "nomn"}, {'VERB', 'past', 'sing'}
GENDERS = ('femn', 'masc')

class phase:
    def __init__(self, message: str):
        self._message = message
    def __enter__(self):
        print(self._message)
        self._start = time()
        return self
    def __exit__(self, type, value, traceback):
        elapsed = time() - self._start
        print(f"Done in {elapsed:.4f} seconds")