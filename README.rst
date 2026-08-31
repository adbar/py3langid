=============
``py3langid``
=============


``py3langid`` is a fork of the standalone language identification tool ``langid.py`` by Marco Lui.

Original license: BSD-2-Clause. Fork license: BSD-3-Clause.



Changes in this fork
--------------------

Execution speed has been improved and the code base has been modernized for Python 3.10+:

- Import: Loading the package (``import py3langid``) is about 30% faster
- Startup: Loading the default classification model is 25-30x faster
- Execution: Language detection with ``langid.classify`` is 5-6x faster on paragraphs (less on longer texts)

For implementation details see this blog post: `How to make language detection with langid.py faster <https://adrien.barbaresi.eu/blog/language-detection-langid-py-faster.html>`_.

The fork also ships a retrained model covering **140 languages** (up from 97)
and a fully rewritten, reproducible training pipeline (see `Training a model`_).

For version history see the `changelog <https://github.com/adbar/py3langid/blob/master/HISTORY.rst>`_.


Usage
-----

Drop-in replacement
~~~~~~~~~~~~~~~~~~~


1. Install the package:

   * ``pip3 install py3langid`` (or ``pip`` where applicable)

2. Use it:

   * with Python: ``import py3langid as langid``
   * on the command-line: ``langid``


With Python
~~~~~~~~~~~

Basics:

.. code-block:: python

    >>> import py3langid as langid

    >>> text = 'This text is in English.'
    # identified language and log-probability score
    >>> langid.classify(text)
    ('en', -68.562286)
    # unpack the result tuple in variables
    >>> lang, prob = langid.classify(text)
    # all languages, most likely first
    >>> langid.rank(text)

Input can be ``str`` or UTF-8 ``bytes``; input is NFC-normalized before
classification, and all-uppercase text is case-folded.

More options:

.. code-block:: python

    >>> from py3langid.langid import LanguageIdentifier, MODEL_FILE

    # subset of target languages
    >>> identifier = LanguageIdentifier.from_model_file(MODEL_FILE)
    >>> identifier.set_languages(['de', 'en', 'fr'])
    # this won't work well...
    >>> identifier.classify('这样不好')
    ('de', -99.862480)

    # normalization of probabilities to an interval between 0 and 1
    >>> identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    >>> identifier.classify('This should be enough text.')
    ('en', 0.9944351)
    # confidence is length-aware: short input scores lower on purpose
    >>> identifier.classify('This is a test')
    ('en', 0.6111177)

    # abstention: return ('und', confidence) below a confidence threshold
    >>> identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True,
    ...                                                 min_confidence=0.2)
    >>> identifier.classify('ok')
    ('und', 0.0070422)


On the command-line
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # basic usage with probability normalization
    $ echo "This should be enough text." | langid -n
    ('en', 0.9935993)

    # define a subset of target languages
    $ echo "This won't be recognized properly." | langid -n -l fr,it,tr
    ('fr', 0.4838270)

Run ``langid`` without input to get an interactive prompt, pipe text into it
to classify a whole document, or add ``--line`` to classify each line
separately. ``langid -u URL`` downloads and classifies a web page. See
``langid --help`` for all options.


Languages
---------

The shipped model knows 140 languages (ISO 639 codes)::

    ace, af, am, an, ar, ary, arz, as, az, ba, bcl, be, bg, bn, br, bs, ca,
    crh, cs, cy, da, de, dz, el, en, eo, es, et, eu, ext, fa, fi, fo, fr,
    fuv, fy, ga, gcf, gcr, gd, gl, gom, grc, gu, gug, guw, ha, hbo, he, hi,
    hr, ht, hu, hy, id, ig, is, it, ja, jv, ka, kab, kik, kk, km, kn, ko,
    ku, ky, la, lb, lg, lij, ln, lo, lt, ltg, lv, mg, mk, ml, mn, mr, ms,
    mt, my, ne, nl, nn, no, nso, oc, om, or, pa, pcm, pl, ps, pt, qu, ro,
    ru, rw, sa, sdh, se, si, sk, sl, sn, so, sq, sr, st, sv, sw, ta, te, tg,
    th, tk, tl, tr, tt, ug, uk, ur, uz, uzs, vec, vi, vo, wa, wuu, xh, yo,
    yue, zh, zu, zxx

``zxx`` is a synthetic "not a language" class that catches numbers, markup,
identifiers, and similar non-linguistic content. With ``min_confidence``
set, low-confidence predictions are returned as ``und`` (undetermined).


Batch mode
----------

``langid -b`` reads file paths from ``stdin`` (one per line) and classifies
the files in parallel, writing CSV to ``stdout``:

.. code-block:: bash

    $ find corpus -name "*.txt" | langid -b
    corpus/a.txt,en,-127.32
    corpus/b.txt,de,-81.15

With ``-d``, the output is one CSV row per file with the full score
distribution over all languages (one column per language).


Web service
-----------

``langid -s`` serves language identification over HTTP (built-in
``wsgiref`` server, default port 9008; ``-r`` binds to the external IP).
The endpoints ``/detect`` and ``/rank`` accept GET (``?q=...``), POST
(``q=...`` form data, or the raw body if no ``q`` key is present), and PUT
(raw body). Responses are JSON:

.. code-block:: bash

    $ curl -d "q=This is a test" localhost:9008/detect
    {"responseData": {"language": "en", "confidence": -46.84}, "responseStatus": 200, "responseDetails": null}

    $ curl -T document.txt localhost:9008/detect

To deploy under a production WSGI server, use the application at
``py3langid.server:application``.


Custom models
-------------

``langid -m FILE`` (or ``LanguageIdentifier.from_modelpath(path)``) loads a
model trained with this package (``model.npz.xz``, a NumPy archive inside
an LZMA stream — no pickle). Models from the original ``langid.py`` are not
supported.


Training a model
----------------

The training pipeline is part of the package:

.. code-block:: bash

    $ python -m py3langid.train.train -m model_dir corpus_dir

Corpus layout, data gathering, corpus hygiene tools, and the design of the
pipeline are documented in `TRAINING.md <https://github.com/adbar/py3langid/blob/master/TRAINING.md>`_.
Training is deterministic: the same corpus and settings reproduce the
model byte for byte.


Read more
---------

``py3langid`` is based on published research. [1] describes the LD feature
selection technique in detail, and [2] presents the original ``langid.py``
tool.

[1] Lui, Marco and Timothy Baldwin (2011) Cross-domain Feature Selection for Language Identification,
In Proceedings of the Fifth International Joint Conference on Natural Language Processing (IJCNLP 2011),
Chiang Mai, Thailand, pp. 553—561. Available from http://www.aclweb.org/anthology/I11-1062

[2] Lui, Marco and Timothy Baldwin (2012) langid.py: An Off-the-shelf Language Identification Tool,
In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL 2012),
Demo Session, Jeju, Republic of Korea. Available from www.aclweb.org/anthology/P12-3005
