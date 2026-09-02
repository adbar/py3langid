=============
``py3langid``
=============


``py3langid`` is a fork of the standalone language identification tool ``langid.py`` by Marco Lui.

Original license: BSD-2-Clause. Fork license: BSD-3-Clause.



Changes in this fork
--------------------

Execution speed has been improved and the code base has been modernized for Python 3.10+:

- Import: Loading the package (``import py3langid``) is about 25% faster
- Execution: Language detection with ``langid.classify`` is 10x faster on single sentences and 3-4x faster on paragraphs (less on longer texts, about 1.4x at 100 kB)
- Startup: Loading the default classification model is 2-3x faster, with a model six times larger

For implementation details see this blog post: `How to make language detection with langid.py faster <https://adrien.barbaresi.eu/blog/language-detection-langid-py-faster.html>`_.

The fork also ships a retrained model covering **139 languages** (up from 97)
and a fully rewritten, reproducible training pipeline (see `Training a model`_).

For version history see the `changelog <https://github.com/adbar/py3langid/blob/master/HISTORY.rst>`_.


Usage
-----

Install: ``pip install py3langid`` — use as ``import py3langid as langid``
or on the command-line as ``langid``.

With Python
~~~~~~~~~~~

.. code-block:: python

    >>> import py3langid as langid

    >>> langid.classify('This text is in English.')
    ('en', -68.562286)
    >>> langid.rank('This text is in English.')   # all languages, most likely first

    >>> from py3langid.langid import LanguageIdentifier, MODEL_FILE
    >>> identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    >>> identifier.set_languages(['de', 'en', 'fr'])
    >>> identifier.classify('This should be enough text.')
    ('en', 0.9999628)

    # abstention: return ('und', confidence) below a threshold
    >>> identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True,
    ...                                                 min_confidence=0.2)
    >>> identifier.classify('ok')
    ('und', 0.0140845)

Input can be ``str`` or UTF-8 ``bytes``; input is NFC-normalized before
classification, and all-uppercase text is case-folded.


On the command-line
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # basic usage with probability normalization
    $ echo "This should be enough text." | langid -n
    ('en', 0.9935992)

    # define a subset of target languages
    $ echo "This won't be recognized properly." | langid -n -l fr,it,tr
    ('fr', 0.4838270)

Run ``langid`` without input to get an interactive prompt, pipe text into it
to classify a whole document, or add ``--line`` to classify each line
separately. ``langid -u URL`` downloads and classifies a web page. See
``langid --help`` for all options.


Languages
---------

The shipped model knows 139 languages plus ``zxx`` (ISO 639 codes)::

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

``langid -s`` serves language identification over HTTP (default port 9008).
Use the ``langid`` console script; ``python -m py3langid.langid`` is not
supported.
Endpoints ``/detect`` and ``/rank`` accept GET, POST, and PUT:

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

``python -m py3langid.train.train -m model_dir corpus_dir``, run from a
clone of the repository (the training code is not part of the PyPI
package) — see
`TRAINING.md <https://github.com/adbar/py3langid/blob/master/TRAINING.md>`_
for corpus layout, data gathering, hygiene, and pipeline design. Training
is deterministic: the same corpus and settings reproduce the model byte for
byte.


Read more
---------

| [1] Lui & Baldwin (2011) `Cross-domain Feature Selection for Language Identification <http://www.aclweb.org/anthology/I11-1062>`_, IJCNLP 2011.
| [2] Lui & Baldwin (2012) `langid.py: An Off-the-shelf Language Identification Tool <http://www.aclweb.org/anthology/P12-3005>`_, ACL 2012 Demo.
