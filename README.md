# KG population API #

### What is this repository for? ###

This repo is meant to demo how the distant supervision module can be used in conjunction with entity and relation extraction models. 
Therefore, this repo contains "frozen" copies of the separate distant supervision repo (https://github.com/Poezedoez/DistantSupervisor) and the entity and relation extraction repos.

We show how to:

1) Annotate documents using distant supervision (https://github.com/Poezedoez/DistantSupervisor)
2) Train an extractor model (includes span-based extraction with SpERT variants and k-nn (NearestNeighBERT))
  * span-based: https://github.com/Poezedoez/span-based-extractors, adaptations from https://github.com/markus-eberts/spert
  * k-NN based: https://github.com/Poezedoez/NearestNeighBERT
3) After loading a trained model, perform inference on documents to extract entities (and relations)

### Configs ###
In the configuration file you can set module specific hyperparameters such as epochs, _k_ in k-NN etc.
Same for the distant supervision module: choose label strategy and things as similarity threshold _cos\_theta_ in the config file.

Default parameters, consisting of all the paths are expected as input in the functions and are not read from the config file.

### Demo ###

* Includes python demo, and command line demo. Uses three zeta objects as data, and an example document.

``` 
python demo.py 
```

or

```
chmod +x demo.sh

./demo.sh
```

* Note that the extraction results, especially from the parametric extractor, are nonsensical when using only the given documents. 
