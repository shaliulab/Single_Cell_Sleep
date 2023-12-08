Sleep models pipeline

Requirements:
===================


`sleep-models==1.1.1`

Instructions
==========================

This pipeline has 5 steps:

* 1: make_dataset. Creates a cached `anndata.AnnData` (.h5ad) only with cells from the desired background.
* These backgrounds are read from the config, and the annotation used is defined in the `backgrounds` folder inside `config["data_dir"]`

2: get_marker_genes: Iteratively removes markers genes with a fold change above an increasingly low threshold, so the cell types
in the background become as homogeneous as possible. A Dimensionality Reduction plot is generated at each threshold

3: remove_marker_genes: Set the genes with a threshold higher than the selected threshold to 0

4: train_models: have one or more of the supported models train using the transcriptomic data to predict sleep and wake

5: predict: predict the sleep / wake status of a cell from the same cell type, or a different one

