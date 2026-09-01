=======
History
=======

Unreleased
----------

* New model covering 139 languages (was 97), retrained from a revamped,
  reproducible training pipeline (``py3langid.train``)
* Breaking: legacy pickled model loading removed, ``npz``+LZMA is the only
  model format; older external models are rejected
* Breaking: WSGI service moved from ``py3langid.langid:application`` to
  ``py3langid.server:application``
* Breaking: an unusable ``-m`` model path now raises instead of silently
  falling back to the bundled model
* Breaking: batch ``--dist`` CSV output gained a ``language`` column
* New ``min_confidence`` option: ``classify()`` returns ``("und", conf)``
  below the threshold (requires ``norm_probs=True``)
* Confidence normalization calibrated across input lengths
* Input normalization matches training (NFC, byte input truncated
  mid-codepoint handled)
* Faster and simplified inference

0.3.0
-----

* Modernized setup, dropped support for Python 3.6 & 3.7
* Simplified inference code
* Support for Numpy 2.0


0.2.2
-----

* Fixed bug in probability normalization (#6)
* Fully implemented data type argument in ``classify()``
* Adapted training scripts to Python3 (untested)


0.2.1
-----

* Maintenance: update and simplify code


0.2.0
-----

* Change Numpy data type for features (``uint32`` → ``uint16``)
* Code cleaning


0.1.2
-----

* Include data in non-wheel package versions


0.1.1
-----

* Faster module loading
* Extended tests and readme


0.1.0
-----

* Fork re-packaged
* Efficiency improvements in ``langid.py``
