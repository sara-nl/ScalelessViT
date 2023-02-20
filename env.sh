#!/bin/bash
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0
module load matplotlib/3.5.2-foss-2022a
module load h5py/3.7.0-foss-2022a

# # had to install typing-extensions from eb
# module load eb/4.7.0
# eblocalinstall --robot /sw/noarch/Debian10/2022/software/EasyBuild/4.7.0/easybuild/easyconfigs/t/typing-extensions/typing-extensions-4.3.0-GCCcore-11.3.0.eb
# export MODULEPATH=${HOME}/.local/easybuild/Debian10/2022/modulefiles:$MODULEPATH
# module load all/typing-extensions/4.3.0-GCCcore-11.3.0

# # virtualev
# source ${HOME}/ScalelessViT/svit/bin/activate
