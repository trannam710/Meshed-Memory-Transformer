import contextlib, sys

class DummyFile(object):
    def flush(self) : pass
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner

def get_tokenizer(tokenizer):
    if callable(tokenizer):
        return tokenizer
    if tokenizer == "vncorenlp":
        try:
            from vncorenlp import VnCoreNLP
            vncorenlp_file = './data/VnCoreNLP/VnCoreNLP-1.1.1.jar'
            tokenizer = VnCoreNLP(vncorenlp_file, annotators="wseg", max_heap_size='-Xmx500m') 
            tokenizer = tokenizer.tokenize
            return tokenizer
        except ImportError:
            print("Please install VNCoreNLP. "
                  "See the docs at github for more information.")
            raise
        except LookupError:
            print("Please install the necessary VNCoreNLP corpora. "
                  "See the docs at  for more information.")
            raise
    
    raise ValueError("Requested tokenizer {}, valid choices are a "
                     "callable that takes a single string as input, "
                     "\"revtok\" for the revtok reversible tokenizer, "
                     "\"subword\" for the revtok caps-aware tokenizer, "
                     "\"spacy\" for the SpaCy English tokenizer, or "
                     "\"moses\" for the NLTK port of the Moses tokenization "
                     "script.".format(tokenizer))
