Linter
======

The :obj:`sdr` library uses `pylint <https://pylint.org/>`_ for static analysis and code
formatting.

Install
-------

First, `pylint` needs to be installed on your system. Easily install it by installing the development dependencies.

.. code-block:: console

   $ python3 -m pip install -r requirements-dev.txt

Configuration
-------------

Various nuisance `pylint` warnings are added to an ignore list in `pyproject.toml`.

.. literalinclude:: ../../pyproject.toml
   :caption: pyproject.toml
   :start-at: [tool.pylint]
   :end-before: [tool.black]
   :language: toml

Run from the command line
-------------------------

Run the linter manually from the command line.

.. code-block:: console

   $ python3 -m pylint src/sdr/

Run from VS Code
----------------

Included is a VS Code configuration file `.vscode/settings.json`.
This instructs VS Code about how to invoke `pylint`.
VS Code will run the linter as you view and edit files.
