l3wrapper
=========

|pypi badge| |docs badge| |doi badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/l3wrapper.svg
    :target: https://pypi.python.org/pypi/l3wrapper
    :alt: Latest PyPI version

.. |Docs Badge| image:: https://readthedocs.org/projects/l3wrapper/badge/
    :alt: Documentation Status
    :scale: 100%
    :target: http://l3wrapper.readthedocs.io

.. |doi badge| image:: https://zenodo.org/badge/244676535.svg
   :target: https://zenodo.org/badge/latestdoi/244676535

.. .. image:: https://travis-ci.org/borntyping/cookiecutter-pypackage-minimal.png
..    :target: https://travis-ci.org/borntyping/cookiecutter-pypackage-minimal
..    :alt: Latest Travis CI build status

A Python 3 wrapper around Live-and-Let-Live (:math:`L^3`) classifier binaries implementing the ``scikit-learn`` estimator interface. The associative classifier was originally published in [#]_.

When imported, the package looks for :math:`L^3` compiled binaries in the user's ``$HOME`` directory. If they are not found, it downloads them.
If you mind letting the wrapper do this for you, you can download the binaries for `macOS Catalina <https://dbdmg.polito.it/wordpress/wp-content/uploads/2020/02/L3C_osx1015.zip>`_ or `Ubuntu 18.04 <https://dbdmg.polito.it/wordpress/wp-content/uploads/2020/03/L3C_ubuntu1804.zip>`_.


.. [#] Elena Baralis, Silvia Chiusano, and Paolo Garza. 2008. A Lazy Approach to Associative Classification. IEEE Trans. Knowl. Data Eng. 20, 2 (2008), 156–171. https://doi.org/10.1109/TKDE.2007.190677

Installation
------------
Install using `pip <http://www.pip-installer.org/en/latest/>`__ with:

::

    pip install l3wrapper

Or, `download a wheel or source archive from
PyPI <https://pypi.python.org/pypi/l3wrapper>`__.

Requirements
^^^^^^^^^^^^

The package is dependent on ``numpy``, ``scikit-learn``, ``tqdm``, and ``requests``.


Usage
-----
By design, the classifier **is intended for categorical/discrete attributes**. Therefore, using subtypes of ``numpy.number`` to fit the model is not allowed.

Simple classification
^^^^^^^^^^^^^^^^^^^^^

A sample usage with the `Car Evaluation dataset <https://archive.ics.uci.edu/ml/datasets/Car+Evaluation>`_:

>>> from l3wrapper.l3wrapper import L3Classifier
>>> import numpy as np
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import accuracy_score
>>> X = np.loadtxt('car.data', dtype=object, delimiter=',')
>>> y = X[:, -1]
>>> X = X[:, :-1]
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
>>> clf = L3Classifier().fit(X_train, y_train)
>>> accuracy_score(y_test, clf.predict(X_test))
0.9071803852889667

Column names and interpretable rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the ``column_names`` and ``save_human_readable`` parameters to obtain an interpretable representation of the model:

>>> column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
>>> clf = L3Classifier().fit(X_train, y_train, column_names=column_names, save_human_readable=True)

The snippet will generate the *level1* and *level2* rule sets. An excerpt is:

::

    0 persons:4,safety:high,maint:low,buying:high acc 12 100.0 4
    1 doors:2,buying:vhigh,safety:med,lug_boot:med unacc 11 100.0 4

in the form::

    <rule_id>\t<antecedent>\t<class label>\t<support count>\t<confidence(%)>\t<rule length>


Known limitations
-----------------

- **fixed** *The parallel training of multiple models cause failures (e.g. using ``GridSearchCV``, ``joblib`` or custom parallelism through ``multiprocessing`` with ``njobs>1``).*
- The scikit-learn's utility ``check_estimator`` still doesn't work, as
  L3Classifier doesn't support numerical input.

Compatibility
-------------

The underlying :math:`L^3` binaries are currently available for **macOS and Ubuntu**.

The package is currently tested with Python 3.6+.

License
-------

The MIT `License <https://github.com/g8a9/l3wrapper/blob/master/LICENSE>`_.

Authors
-------

`l3wrapper` was written by `g8a9 <giuseppe.attanasio@polito.it>`_.
