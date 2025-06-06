#!/bin/bash

source /home/$(whoami)/miniconda3/bin/activate llm_sp

# Shell (bash)
conda install -c conda-forge bash -y

# C++
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y   # on Linux
# (on macOS use clang; on Windows use m2w64-toolchain)

# Java
conda install -c conda-forge openjdk -y

# C# (via Mono)
conda install -c conda-forge mono -y

# PHP
conda install -c conda-forge php -y

# TypeScript (and its dependency, Node.js)
conda install -c conda-forge nodejs typescript -y

# JavaScript (Node.js)
conda install -c conda-forge nodejs -y
