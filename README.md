# Stability and Generalization of Bilevel Programming in Hyperparameter Optimization
Codes for **Stability and Generalization of Bilevel Programming in Hyperparameter Optimization**.

## Requirements
The codes are implemented on
* Python=3.7
* PyTorch=1.4.0

## Run codes


```
python fl_omniglot.py  # run feature learning experiments
```

```
python rw_mnist.py  # run data reweighting experiments
```

The experiment to run is controlled by the `tag` variable in the above `*.py` files. For instance,
* `tag = ablo_K` will run the UD algorithm with different `K`
* `tag = rs_K` will run the random search algorithm with different `K`
* `tag = wdh` will run the UD algorithm with different weight decay in the outer level
* `tag = wdl` will run the UD algorithm with different weight decay in the inner level

You can also specify other settings by modify the `args` variable in the above `*.py` files.

