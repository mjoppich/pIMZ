pIMZ
==========================

This package is still in development. It's not (yet) intended to be used and the pip install may go totally wrong!

Common development tasks
------------------------

  * **Setting up the development environment before first use**
  
        > python3 setup.py build
       
  * **Running tests**  
    Tests are kept in the `tests` directory and are run using

        > py.test
    
  * **Creating Sphinx documentation**
  
        sphinx-quickstart
        (Fill in the values, edit documentation, add it to version control)
        (Generate documentation by something like "cd docs; make html")
        
    (See [this guide](http://sphinx-doc.org/tutorial.html) for more details)
    

  * **Producing executable scripts**  
    Edit the `console_scripts` section of `entry_points` in `src/pIMZ/setup.py`. Then run `buildout`. The corresponding scripts will be created in the `bin/` subdirectory. Note that the boilerplate project already contains one dummy script as an example.

  * **Debugging the code manually**      
    Simply run `bin/python`. This generated interpreter script has the project package included in the path.
    
  * **Publishing the package on Pypi**
  
         > cd src/pIMZ
         > python setup.py register sdist upload
       
  * **Creating an egg or a windows installer for the package**
  
         > cd src/pIMZ
         > python setup.py bdist_egg
          or
         > python setup.py bdist_wininst
       
  * **Travis-CI integration**  
    To use the [Travis-CI](https://travis-ci.org/) continuous integration service, follow the instructions at the [Travis-CI website](https://travis-ci.org/) to register an account and connect your Github repository to Travis. The boilerplate code contains a minimal `.travis.yml` configuration file that might help you get started.

  * **Other tools**  
    The initial `buildout.cfg` includes several useful code-checking tools under the `[tools]` section. Adapt this list to your needs (remember to run `buildout` each time you change `buildout.cfg`).

  * **Working with setup.py**  
    If you are working on a small project you might prefer to drop the whole `buildout` business completely and only work from within the package directory (i.e. make `src\pIMZ` your project root). In this case you should know that you can use
    
         > python setup.py develop
         
    to include the package into the system-wide Python path. Once this is done, you can run tests via
    
         > python setup.py test
         
    Finally, to remove the package from the system-wide Python path, run:
    
         > python setup.py develop -u

  * **Developing multi-package projects**  
    Sometimes you might need to split your project into several packages, or use a customized version of some package in your project. In this case, put additional packages as subdirectories of `src/` alongside the original `src/pIMZ`, and register them in `buildout.cfg`. For example, if you want to add a new package to your project, do:
    
         > cd src/
         > cookiecutter https://github.com/audreyr/cookiecutter-pypackage.git
           or
         > paster create <new_package_name>
         
    Then add `src/<new_package_name>` to version control and add the directory `src/<new_package_name>` to the `develop` list in `buildout.cfg`. Also, if necessary, add `<new_package_name>` to the `[main]` part of `buildout.cfg` and mention it in the `[pytest]` configuration section of `setup.cfg`.

Copyright & License
-------------------

  * Copyright 2019, Markus Joppich
  * License: MIT
