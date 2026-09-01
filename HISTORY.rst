=======
History
=======

0.4.0
-----

* New model: 139 languages + ``zxx`` (was 97)
* Reproducible training pipeline (``py3langid.train``)
* ``min_confidence`` option: return ``und`` below threshold
* Length-calibrated confidence normalization
* Input normalized as in training (NFC)
* Faster inference
* Breaking: ``nb`` merged into ``no``; ``set_languages(["nb"])`` raises
* Breaking: ``npz``+LZMA is the only model format (pickle removed)
* Breaking: WSGI moved to ``py3langid.server:application``
* Breaking: invalid ``-m`` path raises instead of silent fallback
* Breaking: ``--dist`` CSV gained a ``language`` column

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
