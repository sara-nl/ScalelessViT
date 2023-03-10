# Building

To build the program run the following commands.

```shell
source ../env.sh

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.
make
make install
```

This will create a subdirectory called `bin` in your build directory which will contain the executable you can use to
run and test any implemented algorithms.

# Running

Run the program, we do a dirty hack and add the current build dir to the library path so that we do not move our testing
library anywhere we do not want it to.

```shell
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:." ./bin/crop_interpolate
```

# Testing

If everything works correctly, you should have output like this:

```text
crop_interpolate_default (cpu) took 21.1034 ms
call_ci_kernel (cuda:0) took 0.107063 ms
reference == kernel: 0
```

If you implemented your kernel correctly, the final line should be

```text
reference == kernel: 0
```