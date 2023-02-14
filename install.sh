#!/bin/bash

. env.sh
cd cxx || exit
python setup.py install --prefix=~/.local/
