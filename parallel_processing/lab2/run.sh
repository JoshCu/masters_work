#!
/usr/local/cuda/bin/nvcc -I /usr/local/cutil/include/ $1.cu -o $1 -L /usr/local/cutil/lib/ -lcutil_x86_64

