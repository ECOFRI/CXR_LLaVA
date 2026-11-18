"""METEOR from vendored NLTK 3.6.7 scorer and installed NLTK WordNet."""
import hashlib
import importlib.util
from pathlib import Path

from vlm_medical.eval.text_metrics import tokenize

SCORER = Path(__file__).parent / 'vendor/nltk367_meteor_score.py'


def scorer():
    spec = importlib.util.spec_from_file_location('nltk367_meteor_score', str(SCORER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_pairs(pairs):
    module = scorer()
    return [module.single_meteor_score(tokenize(ref), tokenize(hyp)) for ref, hyp in pairs]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
