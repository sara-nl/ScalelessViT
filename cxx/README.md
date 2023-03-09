# Building

To build the program run the following commands.

```shell
source ../env.sh

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_DIR=.
make
make install
```

This will create a subdirectory called `bin` in your build directory which will contain the executable you can use to run and test any implemented algorithms.