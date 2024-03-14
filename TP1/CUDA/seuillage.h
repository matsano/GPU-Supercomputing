
//#define SEUILLAGE_H
//#ifndef SEUILLAGE_H

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>



#define SIZE_I 960
#define SIZE_J 1280


#define BLOCK_SIZE 16

// Prototype
void runTest( int argc, char** argv);
extern "C" void seuillage_C( float reference[][SIZE_J][SIZE_I] , float idata[][SIZE_J][SIZE_I] );


//#endif



