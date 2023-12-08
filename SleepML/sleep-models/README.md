Sleep models
===============

* Repository storing the program used to compare sleep patterns across cell types for Joana D. et al.
* The program is fully written in Python.
* It trains different instances of the same model on different cell types, and then evaluates their performance on all cell types.
* The idea being to show that cells sleep differently.
* The model is a binary classifier that predicts whether any given cell comes from a sleeping or awake fly.
* The model architectures:
    * Explainable Boosting Machine (EBM), a so-called 'glassbox' model with straightforward interpretability and still highly capable of learning complex functions mapping input to output. This model is implemented in the interpretml Python package.
    * Multilayer Perceptron Model, a classic vanilla network


Vocabulary
================

**features**: Numerical data representing gene expression of a single cell. Each gene is mapped to a column, and each row to a cell
**labels**: String data representing the category a given cell belongs to, according to the original experimental design. Possible categories are Treatment, Condition, Sleep Stage, Genotype, Run
**pseudo-labels**: String data representing a new category produced by merging several label categories. The merging pattern is made explicit in a template file in .yaml format, like so

```
ZT 8 gab: "drug"
ZT 20 sleep: "sleep"
ZT 20 midline cross: "sleep"
ZT 14 sleep: "sleep"
ZT 14 SD: "SD"
ZT 20 sleep deprivation: "SD"
ZT 2 sleep drive 14h SD: "SD++"
```

in this file, each condition is a label, and `drug`, `sleep`, `SD` and `SD++` are pseudo-labels produced by merging several conditions

Labels and pseudolabels should have only one column and as many rows as cells

**code**: Ordinal data representing labels


Organization
===============

## Preprocessing

Logic to filter and split the input adata so the "experimental design" is recapitulated by the program.

Exposed under the `make-dataset` entrypoint.

## Train

Logic to train an model with a given cell type

Exposed under the `train-model` entrypoint.

The models live in the models module. The following are available:

Regression:

* Multilayer Perceptron (sklearn.neural_network.MLPRegressor)

Classification

* Explainable Boosting Machine (interpret.glassbox.ExplainableBoostingClassifier)


## Predict

Logic to predict the sleep/wake status of a cell or group of cells on a previously trained EBM model

Exposed under the `crosspredict` entrypoint.

## Plotting

Logic to generate the matrixplot

Exposed under the `make-matrixplot` entrypoint.


Execution
================

Sleep Classifier comparison
==================================

This repository contains the code needed to perform a comparison of Sleep vs Wake classifier accuracy across a selected list of cell types.

The classifier is a Explainable Boosting Machine (EBM) a glassbox model with human friendly interpretability.

The data was taken from vsc `/staging/leuven/stg_00003/cbd-bioinf/CBD__SHLI__Joana_Dopp__Sleep_Signature/NovaSeq6000_20201130/20201214/Preloom/`.

## To generate the glia, KC cell, etc specific datasets:

Call `make-dataset`

Example for KC cells:

```
make-dataset --h5ad-input data/h5ad/Preloom/All_Combined_No_ZT2_Wake.h5ad  --random_state 1500 --background "data/backgrounds/KC_mapping.csv" --batch-genes-file  "data/batch_effects.xlsx" --shuffles 6
```

Explanation:

* The `--h5ad-input` argument points to the source h5ad that contains all data for all cells (not just KC).
* The `--random_state` argument is an integer that makes the analysis reproducible by removing the randomness. Computational replicates can be obtained by rerunning this and everything downstream with another random_state.
* The `--background` argument is the path to a .csv file with columns `cluster` `louvain_res` and `idx`. Each row states at which louvain resolution the corresponding cluster is defined (and what is the idx it gets under that resolution).
* The `--batch-genes-file` argument points to an excel sheet with batch effects genes that should be removed.
* The `--shuffles` argument is an integer that states how many independent shuffles will be done.
* The `--template-file` argument is a path to a json file stating a mapping between a categorical variable and some meaningful numeric value

The result is a new h5ad file in the same folder as the All_Combined_No_ZT2_Wake.h5ad as well as several shuffled h5ad files, also in the same folder. The h5ad files contain cells only belonging to the passed background. And the resulting adata comes with a new column "CellType" which is set to the corresponding cluster of that cell.

An option to shuffle the data is available. This is useful when simulating the null hypothesis e.g. showing that the better performance achieved when predicting on the same cell type on which it was trained is not due to simple overfitting (even if performance is evaluated on a test set, this is still possible). If a model trained on a random group of cells cannot predict well on a test set, but it predicts well on a well defined group of cells, that strongly suggests 1) the group of cells have some inherent properties and 2) the model is learning them!



# To find the marker genes of a cluster:

Call `get-marker-genes`



## To train the cell specific classifiers:


Call `train-model`

```
train-model  $CLUSTER --h5ad-input $H5AD_INPUT --background $BACKGROUND --exclude-genes-file $EXCLUDE --results $RESULTS --random_state $random_state --verbose 20 --logfile logs/$LOGFILE
```

* The first argument is positional and it should be a character with the name of a cluster as annotated in the obs table of the original h5ad, and it should be included in the inputting h5ad. It should also match one of the clusters in the background
* `--background` is identical to above
* `--exclude-genes-file` is a txt with one gene per line. These genes are not used in the training
* `--results` folder to which the output of the program is saved. If it does not exist, the program creates it on the spot.
* `--random_state` is identical to above
* `--verbose` level of verbosity (less is more)
* `--logfile` path to a log file where program execution information is saved


## To compare classifier performance and plot it:

Call `crosspredict`

```
crosspredict --background $BACKGROUND --root-dir $ROOT_DIR --output $OUTPUT --ncores $NCORES
```

* `--background` is identical to above.
* `--root-dir` is the path to one of the folders created in the `train-model` step.
* `--output` is the path to the resulting png file
* `--ncores` is an integer stating the number of cores to use (use more for faster analysis)

