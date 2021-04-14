# deadtrees

Selinas Dead Tree Mapping project 🌲💀🌲 ...

To install the package and the required dependencies:

```
pip install -e . 
```

If you also want to train the model install the extra package dependencies for subplacke `train`:

```
pip install -e ".[train]"
```

If you want to rerun preprocessing instead of reading the data from the S3 object-storage install the `preprocess` subpackage dependencies:

```
pip install -e ".[preprocess]"
```

Alternatively, install all subpackages with:

```
pip install -e ".[all]"
```