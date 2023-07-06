#!/bin/bash

mkdir -p tmp
pushd tmp
    # Regression
    wget -c https://github.com/GuiArcencio/metal-crats-files/raw/main/regression_datasets.tar.gz
    tar -xzvf regression_datasets.tar.gz
    mv regression_datasets/* ../assets/regression/datasets
popd

rm -rf tmp