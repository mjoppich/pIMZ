#ifndef SRM_H
#define SRM_H

#ifdef _MSC_VER

typedef __int8 int8_t;
typedef unsigned __int8 uint8_t;

typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;

#else
#include <stdint.h>
#endif

#include <vector>
#include <math.h>
#include <limits>
#include <algorithm>
#include <map>
#include <iostream>
#include "imageregion.h"

#ifndef  SAFEDEL
#define SAFEDEL(x) {if (x != NULL) {delete x; x=NULL;}}
#endif

enum SEGMODE { EUCL, DOTPROD};

class PixelRegion
{
public:
    
    PixelRegion(uint32_t iIndex, uint32_t xIndex, uint32_t yIndex, ImageRegion* pRegion)
    {
        m_iXIndex = xIndex;
        m_iYIndex = yIndex;
        m_iIndex = iIndex;

        m_pRegion = pRegion;
    }

    uint32_t getIndex()
    {
        return m_iIndex;
    }

    uint32_t getXIndex()
    {
        return m_iXIndex;
    }
    uint32_t getYIndex()
    {
        return m_iYIndex;
    }

    ImageRegion *getRegion()
    {
        return m_pRegion;
    }

    void updateRegion( ImageRegion *pRegion)
    {
        m_pRegion = pRegion;
    }


private:
    uint32_t m_iXIndex, m_iYIndex, m_iIndex;
    ImageRegion* m_pRegion;
};


class SRM
{
public:
    SRM(uint32_t iDims, float* pQValues, uint8_t iQCount);
    ~SRM();
    void setParameters();

    uint32_t* getSegmentedImageFloat(uint32_t xcount, uint32_t ycount, float* pImage);
    float* calculateSimilarity(uint32_t xcount, uint32_t ycount, float* pImage);


    void setSegmentMode(SEGMODE emode)
    {
        m_eSegmentMode = emode;
    }

private:

    static bool fequals(float f1, float f2);

    static float dotProduct(float* pData1, float* pData2, uint32_t icount, bool verbose=false);

    bool testMergeRegions(PixelRegion *pR1, PixelRegion *pR2, float fQ, uint32_t iImageSize, std::map<int, int>* pRegionsOfCardinality);
    static float distanceFunction(ImageRegion *pR1, ImageRegion *pR2);
    static float distanceFunctionDot(ImageRegion *pR1, ImageRegion *pR2);

    static bool sortGradients(std::pair<PixelRegion *, PixelRegion *> *pP1, std::pair<PixelRegion *, PixelRegion *> *pP2);

    ImageRegion *mergeRegions(ImageRegion *pR1, ImageRegion *pR2, std::map<int, int>* pRegionsOfCardinality);

    float getB(ImageRegion *pRegion, float fQ, uint32_t iImageSize, std::map<int, int>* pRegionsOfCardinality);

    uint32_t m_iDims = 0;
    std::vector<float> m_vQs;

    SEGMODE m_eSegmentMode = SEGMODE::EUCL;

};

#endif // IGEM_SRM_H
