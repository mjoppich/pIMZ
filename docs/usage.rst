==========================================
The pIMZ framework: installation and usage
==========================================

This section will tell you how to `install <Installation_>`_ pIMZ, but also refer you to some examples on `how to use <Usage>`_ pIMZ.




Installation
------------

The easiest way to install most Python packages is via ``pip``.

If not already done, you must first install the following dependencies manually. This is because at the time of writing this document, dabest requires ``pandas~=0.25``, which is incompatible with probably the rest of the world nowadays. So first dabest is installed, then numpy and pandas are upgraded again ::

    sudo pip3 install dabest
    sudo pip3 install numpy pandas --upgrade

Only then we should install ``pIMZ`` ::

    sudo pip3 install pIMZ

``pIMZ`` is now ready to go!


Usage
-----

pIMZ can be used with several use-cases.
Here two of these are highlighted.

1. Using a local imzML-file as source
2. Using public HuBMap-Data

Local imzML-file
````````````````

The usage of pIMZ on a local imzML is shown in the examples.
The `IMZMLprocess notebook  <https://github.com/mjoppich/pIMZ/blob/master/examples/IMZMLprocess.ipynb>`_ showcases how to use pIMZ to retrieve the correct data, normalize intensities, cluster spectra, determine background, perform differential analysis and finally combine multiple region for further comparative analyses.


Public HuBMap-Data
``````````````````

The HuBMap-Data example is divided into two notebooks.
The first notebook describes the use of the `HuBMap-downloader <https://github.com/mjoppich/pIMZ/blob/master/examples/GlobusTest.ipynb>`_.
Since HuBMap-files are stored on Globus, specific care to access their API has to be taken. 
In order to download files yourself, you have to run `Globus Connect Personal <https://www.globus.org/globus-connect-personal>`_ on the machine where you want to download the data to.

The usage of pIMZ on a public HuBMap-generated tissue is shown in the examples.
The `HuBMAP_kidney notebook <https://github.com/mjoppich/pIMZ/blob/master/examples/HuBMAP_kidney.ipynb>`_ showcases how to use pIMZ to retrieve the correct data, normalize intensities, cluster spectra, determine background, perform differential analysis and finally combine multiple region for further comparative analyses.
In general, the workflow is no different that for any other local data.