python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade twine

# Release idtrackerai
python3 -m build
python3 -m twine upload --verbose dist/* # this requires PyPI user and password
