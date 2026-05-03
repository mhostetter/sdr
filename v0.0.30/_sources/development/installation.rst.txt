Installation
============

Install from PyPI
-----------------

The latest released version of :obj:`sdr` can be installed from `PyPI <https://pypi.org/project/sdr/>`_ using `pip`.

.. code-block:: console

   $ python3 -m pip install sdr

Install from GitHub
-------------------

The latest code on `main` can be installed using `pip` in this way.
The `main` branch should be equivalent to the latest release on PyPI.

.. code-block:: console

   $ python3 -m pip install git+https://github.com/mhostetter/sdr.git

Or a pre-released branch (e.g. `release/0.0.x`) can be installed this way.

.. code-block:: console

   $ python3 -m pip install git+https://github.com/mhostetter/sdr.git@release/0.0.x

Editable install from local folder
----------------------------------

To actively develop the library, it is beneficial to `pip install` the library in an
`editable <https://pip.pypa.io/en/stable/cli/pip_install/?highlight=--editable#editable-installs>`_
fashion from a local folder.
This allows changes in the current directory to be immediately seen upon the next `import sdr`.

Clone the repo wherever you'd like.

.. code-block:: console

    $ git clone https://github.com/mhostetter/sdr.git

Install the local folder using the `-e` or `--editable` flag.

.. code-block:: console

    $ python3 -m pip install -e sdr/

Install the `dev` dependencies
------------------------------

The development dependencies include packages for linting and unit testing.
These dependencies are stored in `requirements-dev.txt`.

.. literalinclude:: ../../requirements-dev.txt
   :caption: requirements-dev.txt

Install the development dependencies by passing `-r` to `pip install`.

.. code-block:: console

   $ python3 -m pip install -r requirements-dev.txt
