// Define C functions for the C++ class - as ctypes can only talk to C...



#include "./src/srm.h"
#include <stdio.h>

extern "C" 
{

    SRM* StatisticalRegionMerging_New( uint32_t const colorDims, float* pQValues, uint8_t iQCount)
    {
        printf("SRM C++ Object here with %i dimensions and %i Qs\n", colorDims, iQCount);
        return new SRM(colorDims, pQValues, iQCount);
    }

    void* StatisticalRegionMerging_mode_dot( SRM* pSRM)
    {
        pSRM->setSegmentMode( SEGMODE::DOTPROD );
    }

    void* StatisticalRegionMerging_mode_eucl( SRM* pSRM)
    {
        pSRM->setSegmentMode( SEGMODE::EUCL );
    }

    float* SRM_calc_similarity( SRM* pSRM, uint32_t xcount, uint32_t ycount, float* pImage)
    {
        
        return pSRM->calculateSimilarity(xcount, ycount, pImage);
    }

    void SRM_test_matrix( SRM* pSRM, uint32_t xcount, uint32_t ycount, float* pImage)
    {

        for (int i = 0; i < xcount*ycount*3; ++i)
        {
            printf("%i %f\n", i, pImage[i]);
        }


    }

    uint32_t* SRM_processFloat(SRM* pSRM, uint32_t xcount, uint32_t ycount, float* pImage)
    {
        return pSRM->getSegmentedImageFloat(xcount, ycount, pImage);
    }

};