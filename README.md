# KG population API #

### What is this repository for? ###

* Annotate documents using distant supervision
* Train an extractor model (includes span-based extraction with SpERT variants and k-nn (NearestNeighBERT))
* After loading a trained model, perform inference on documents to extract entities (and relations)

### Configs ###
In the configuration file you can set module specific hyperparameters such as epochs, _k_ in k-NN etc.
Same for the distant supervision module: choose label strategy and things as similarity threshold _cos\_theta_ in the config file.

Default parameters, consisting of all the paths are expected as input in the functions and are not read from the config file.

### Demo ###

* Includes python demo, and command line demo. Uses three zeta objects as data, and an example document.

``` python demo.py ```

or

```chmod +x demo.sh
./demo.sh
```

* Note that the extraction results, especially from the parametric extractor, are nonsensical when using only the given documents. 