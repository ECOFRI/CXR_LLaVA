"""Restore only public pinned NLP assets; no clinical input or external upload."""
import hashlib
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from research.rrg_f0.core import ROOT, sha256
from research.rrg_f1.cohort import BASE

URLS = {
    'GENIA+PubMed_archive': ('https://www.dropbox.com/s/ev3h78gq7526xdj/BLLIP-GENIA-PubMed.tar.bz2?dl=1', 'parser/GENIA-PubMed.tar.bz2'),
    'CoreNLP3.5.2': ('https://repo.maven.apache.org/maven2/edu/stanford/nlp/stanford-corenlp/3.5.2/stanford-corenlp-3.5.2.jar', 'stanford/stanford-corenlp-3.5.2.jar'),
    'punkt': ('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip', 'nltk/punkt.zip'),
    'universal_tagset': ('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/universal_tagset.zip', 'nltk/universal_tagset.zip'),
    'mkl-2019.1-144.tar.bz2': ('https://repo.anaconda.com/pkgs/main/linux-64/mkl-2019.1-144.tar.bz2', 'legacy_libs/mkl-2019.1-144.tar.bz2'),
    'intel-openmp-2019.1-144.tar.bz2': ('https://repo.anaconda.com/pkgs/main/linux-64/intel-openmp-2019.1-144.tar.bz2', 'legacy_libs/intel-openmp-2019.1-144.tar.bz2'),
    'libffi-3.2.1-he1b5a44_1006.tar.bz2': ('https://conda.anaconda.org/conda-forge/linux-64/libffi-3.2.1-he1b5a44_1006.tar.bz2', 'legacy_libs/libffi-3.2.1-he1b5a44_1006.tar.bz2')
}


def link_if_absent(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def main():
    hashes = json.loads((BASE / 'labeler_assets_summary.json').read_text())['public_assets_sha256']
    cache = ROOT / 'model_cache/rrg_f1'
    for name, (url, relative) in URLS.items():
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.tmp')
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    stream.write(block)
                    h.update(block)
            if h.hexdigest() != hashes[name]:
                raise ValueError('STOP_PUBLIC_ASSET_SHA_MISMATCH')
            temporary.replace(path)
        elif sha256(path) != hashes[name]:
            raise ValueError('STOP_CACHED_PUBLIC_ASSET_SHA_MISMATCH')
    model = cache / 'parser/GENIA+PubMed'
    if not model.exists():
        model.mkdir()
        with tarfile.open(cache / 'parser/GENIA-PubMed.tar.bz2') as archive:
            archive.extractall(model, filter='data')
    local = json.loads((BASE / 'paths.local.json').read_text())
    link_if_absent(Path(local['parser_model_dir']), model)
    for name, kind in [('punkt', 'tokenizers'), ('universal_tagset', 'taggers')]:
        target = cache / 'nltk' / name
        if not target.exists():
            with zipfile.ZipFile(cache / 'nltk' / (name + '.zip')) as archive:
                if any(Path(n).is_absolute() or '..' in Path(n).parts for n in archive.namelist()):
                    raise ValueError('Unexpected public ZIP member')
                archive.extractall(cache / 'nltk')
        link_if_absent(ROOT / 'research/blip2_cxr_v4/.cache/nltk' / kind / name, target)
    for name in ('mkl-2019.1-144.tar.bz2', 'intel-openmp-2019.1-144.tar.bz2', 'libffi-3.2.1-he1b5a44_1006.tar.bz2'):
        with tarfile.open(cache / 'legacy_libs' / name) as archive:
            archive.extractall(cache / 'legacy_libs', members=[m for m in archive.getmembers() if m.name.startswith('lib/')], filter='data')
    jar = cache / 'stanford/stanford-corenlp-3.5.2.jar'
    link_if_absent(Path.home() / '.local/share/pystanforddeps' / jar.name, jar)
    print('Public NLP asset hashes verified; legacy Python environment must already exist')


if __name__ == '__main__':
    main()
