#python3 setup.py register sdist upload

rm -rf dist
python setup.py sdist
twine upload -r testpypi dist/*
#twine upload dist/*