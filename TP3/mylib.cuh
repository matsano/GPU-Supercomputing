#ifndef MYLIB_CUH
#define MYLIB_CUH

#include "mylib.h"


#include <cuda_runtime.h>
#include <math.h>


Mat seuillageGPU( Mat in);

Mat sobelGPU( Mat in);

Mat nbGPU( Mat in);

#endif
