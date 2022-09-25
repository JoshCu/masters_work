nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx86\x64" -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include" blur.cu -l freeglut -o blur.exe

blur.exe