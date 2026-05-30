Unit Tests
==========

The :obj:`sdr` library uses `pytest <https://docs.pytest.org/en/7.0.x/>`_ for unit testing.

Install
-------

First, `pytest` needs to be installed on your system. Easily install it by installing the development dependencies.

.. code-block:: console

   $ python3 -m pip install -r requirements-dev.txt

Configuration
-------------

The `pytest` configuration is stored in `pyproject.toml`.

.. literalinclude:: ../../pyproject.toml
   :caption: pyproject.toml
   :start-at: [tool.pytest.ini_options]
   :end-before: [tool.coverage.report]
   :language: toml

Run from the command line
-------------------------

Execute all of the unit tests manually from the command line.

.. code-block:: console

    $ python3 -m pytest tests/

Or only run a specific test module.

.. code-block:: console

    $ python3 -m pytest tests/modulation/

Or only run a specific unit test file.

.. code-block:: console

    $ python3 -m pytest tests/modulation/test_psk.py

Run from VS Code
----------------

Included is a VS Code configuration file `.vscode/settings.json`.
This instructs VS Code about how to invoke `pytest`.
VS Code's integrated test infrastructure will locate the tests and allow you to run or debug any test.
