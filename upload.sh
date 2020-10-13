#python3 setup.py register sdist upload

rm -rf dist
rm -rf build
rm -rf pIMZ.egg-info

python setup.py sdist
twine upload -r testpypi dist/*
#twine upload dist/*