=======
History
=======

0.5.0 (unreleased)
------------------

* New model: 2-word inputs +8 to +9 points (WiLI, OpenLID), CommonLID +2.3,
  7.2 MB (was 4.6)
* Token table: per-language word and CJK character credits for short input
* Chinese trained as two script classes folded into ``zh``
* Input lowercased and space-padded, as in training
* Featureless input uniform under ``norm_probs`` (was ``sr``-biased)
* Training: one command gathers and cleans the corpus and writes
  ``MANIFEST.json``
* Training: model-free cleaning (``clean``), ``verify`` removed
* Training: cleaner sources (wiki article leads, fewer topup sources, no
  FLORES-200 rows, pinned dataset revisions)

Breaking:

* ``LanguageIdentifier`` takes a ``modelio.Model``. ``nb_ptc``, ``nb_pc``
  and ``nb_classes`` removed (use ``labels``)
* 0.4.0 models rejected, retrain with ``py3langid.train.train``
* ``modelio.expand_nextmove`` and CLI ``-u/--url``, ``-r/--remote`` removed
* ``sdh`` dropped (138 languages)

0.4.0
-----

* New model: 139 languages + ``zxx`` (was 97)
* Reproducible training pipeline (``py3langid.train``)
* Faster inference; input NFC-normalized as in training
* Length-calibrated confidence normalization, with ``min_confidence``
  to return ``und`` below a threshold

Breaking:

* ``nb`` merged into ``no``; ``set_languages(["nb"])`` raises
* ``npz``\ +LZMA is the only model format (pickle removed)
* WSGI moved to ``py3langid.server:application``
* invalid ``-m`` path raises instead of silent fallback
* ``--dist`` CSV gained a ``language`` column
* ``cl_path``/``rank_path`` removed, module-level and on
  ``LanguageIdentifier`` (read the file and call ``classify``/``rank``)

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
