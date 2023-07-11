#!/bin/bash

mkdir -p tmp
pushd tmp
    # Regression
    gdown 1Rd_Qm5jG9jTSXBK0VqS7iXrjSBIxsaFC
    tar -xzvf regression_datasets.tar.gz
    mkdir -p ../assets/regression/datasets
    mv regression_datasets/* ../assets/regression/datasets

    # Classification
    gdown 1k8m1BU7XPE5OHzyD_evrS8gYsFMcuPci
    tar -xzvf classification_datasets.tar.gz
    mkdir -p ../assets/classification/datasets
    mv classification_datasets/* ../assets/classification/datasets
popd

rm -rf tmp