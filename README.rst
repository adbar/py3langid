=============
``py3langid``
=============


``py3langid`` is a fork of the standalone language identification tool ``langid.py`` by Marco Lui.

It is a Naive Bayes classifier:

- Byte n-gram features are selected by distinctiveness
- Input bytes are scanned with a DFA to count features
- Scoring is a single matrix product against per-class log probabilities
- A table of per-language word credits helps on short input

Original license: BSD-2-Clause. Fork license: BSD-3-Clause.



Changes in this fork
--------------------

The code base has been modernized for Python 3.10+. Compared with ``langid.py``:

- Import: ``import py3langid`` is about 20% faster
- Execution: ``classify`` is 4 to 12x faster on short sentences and 1.5 to 2.5x faster on 300 characters. It breaks even between 1 and 3 kB. It is 1.4 to 2.6x slower at 100 kB
- Startup: the default model loads about 10% faster (0.6 s)

For implementation details see this blog post: `How to make language detection with langid.py faster <https://adrien.barbaresi.eu/blog/language-detection-langid-py-faster.html>`_.

The fork ships a retrained model covering **138 languages** (up from 97)
and a reproducible training pipeline (see `Training a model`_).
It ranks first on the test set of `CommonLID <https://github.com/commoncrawl/commonlid-eval>`_,
a language identification benchmark built from Common Crawl web text.

For version history see the `changelog <https://github.com/adbar/py3langid/blob/master/HISTORY.rst>`_.


Usage
-----

Install with ``pip install py3langid``. Use it as ``import py3langid as langid``
or on the command line as ``langid``.

With Python
~~~~~~~~~~~

.. code-block:: python

    >>> import py3langid as langid

    >>> langid.classify('This text is in English.')
    ('en', -54.686)
    >>> langid.rank('This text is in English.')[:2]   # all languages, most likely first
    [('en', -54.686), ('nl', -115.939)]

    >>> from py3langid.langid import LanguageIdentifier, MODEL_FILE
    >>> identifier = LanguageIdentifier.from_modelpath(MODEL_FILE, norm_probs=True)
    >>> identifier.set_languages(['de', 'en', 'fr'])
    >>> identifier.classify('This should be enough text.')
    ('en', 1.0)

    # abstention: return ('und', confidence) below a threshold
    >>> identifier = LanguageIdentifier.from_modelpath(MODEL_FILE, norm_probs=True,
    ...                                                min_confidence=0.5)
    >>> identifier.classify('ovo je test')
    ('und', 0.432)

Input can be ``str`` or UTF-8 ``bytes``. It is lowercased and NFC-normalized
before classification.


On the command-line
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # basic usage with probability normalization
    $ echo "This should be enough text." | langid -n
    ('en', 1.0)

    # define a subset of target languages
    $ echo "This won't be recognized properly." | langid -n -l fr,it,tr
    ('it', 0.5943993330001831)

Run ``langid`` without input to get an interactive prompt, pipe text into it
to classify a whole document, or add ``--line`` to classify each line
separately. See ``langid --help`` for all options.


Languages
---------

The shipped model knows 138 languages plus ``zxx`` (ISO 639 codes)::

    ace, af, am, an, ar, ary, arz, as, az, ba, bcl, be, bg, bn, br, bs, ca,
    crh, cs, cy, da, de, dz, el, en, eo, es, et, eu, ext, fa, fi, fo, fr,
    fuv, fy, ga, gcf, gcr, gd, gl, gom, grc, gu, gug, guw, ha, hbo, he, hi,
    hr, ht, hu, hy, id, ig, is, it, ja, jv, ka, kab, kik, kk, km, kn, ko,
    ku, ky, la, lb, lg, lij, ln, lo, lt, ltg, lv, mg, mk, ml, mn, mr, ms,
    mt, my, ne, nl, nn, no, nso, oc, om, or, pa, pcm, pl, ps, pt, qu, ro,
    ru, rw, sa, se, si, sk, sl, sn, so, sq, sr, st, sv, sw, ta, te, tg, th,
    tk, tl, tr, tt, ug, uk, ur, uz, uzs, vec, vi, vo, wa, wuu, xh, yo, yue,
    zh, zu, zxx

``zxx`` is a synthetic "not a language" class that catches numbers, markup,
identifiers, and similar non-linguistic content. With ``min_confidence``
set, low-confidence predictions are returned as ``und`` (undetermined).


Batch mode
----------

``langid -b`` reads file paths from ``stdin`` (one per line) and classifies
the files in parallel, writing CSV to ``stdout``:

.. code-block:: bash

    $ find corpus -name "*.txt" | langid -b
    corpus/a.txt,en,-9078.8720703125
    corpus/b.txt,de,-13344.197265625

With ``-d``, the output is one CSV row per file with the full score
distribution over all languages (one column per language).


Web service
-----------

``langid -s`` serves language identification over HTTP (default port 9008).
Use the ``langid`` console script.
``python -m py3langid.langid`` is not supported.
Endpoints ``/detect`` and ``/rank`` accept GET with a ``q`` query parameter,
POST with a form-encoded ``q`` field (JSON is not accepted),
and PUT with the raw text as body:

.. code-block:: bash

    $ curl -d "q=This is a test" localhost:9008/detect

For production, use ``py3langid.server:application`` under a WSGI server.


Custom models
-------------

``langid -m FILE`` (or ``LanguageIdentifier.from_modelpath(path)``) loads a
model trained with this package (``model.npz.xz``). Models from the
original ``langid.py`` are not supported.


Training a model
----------------

Run ``python -m py3langid.train.train -m model_dir corpus_dir`` from a clone
of the repository, since the training code is not part of the PyPI package.
Training is deterministic: the same corpus and settings reproduce the model
byte for byte. See
`TRAINING.md <https://github.com/adbar/py3langid/blob/master/TRAINING.md>`_
for corpus layout, data gathering and pipeline design.


Read more
---------

| [1] Lui & Baldwin (2011) `Cross-domain Feature Selection for Language Identification <http://www.aclweb.org/anthology/I11-1062>`_, IJCNLP 2011.
| [2] Lui & Baldwin (2012) `langid.py: An Off-the-shelf Language Identification Tool <http://www.aclweb.org/anthology/P12-3005>`_, ACL 2012 Demo.
