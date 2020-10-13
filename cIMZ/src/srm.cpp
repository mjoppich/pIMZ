#include "srm.h"
#include <limits>       // std::numeric_limits
#include <assert.h>
#include <cfloat>
#include <omp.h>

SRM::SRM(uint32_t iDims, float* pQValues, uint8_t iQCount)
{
    m_iDims = iDims;

    for (uint8_t q=0; q < iQCount; ++q)      
    {
        m_vQs.push_back(pQValues[q]);
    }
}

SRM::~SRM()
{

    //delete(m_pSImage);
    //delete(m_pImage);

}

void SRM::setParameters()
{

}

float* SRM::calculateSimilarity(uint32_t xcount, uint32_t ycount, float* pImage)
{
    size_t iNumFields = xcount * ycount * xcount * ycount;
    float *pSim = (float*) malloc(sizeof(float) * iNumFields);

    float* pCurData;
    float* pCompareData;

    uint32_t iElements = xcount * ycount;

    std::cerr << "Element count " << iElements << std::endl;
    
    uint32_t ix = 0;
    uint32_t iCurIndex=0;
    uint32_t iCompareIndex=0;
    uint32_t iCurElement=0;
    uint32_t iCompareElement=0;
    float simValue = 0.0f;

    for (uint8_t i = 0; i < 10; ++i)
    {
        std::cout<< "img i=" << pImage[i] << " ";
    }
    std::cout << std::endl;
    
    //omp_set_num_threads(1);

    /*
    for (uint32_t i=0; i < xcount * ycount * m_iDims; ++i)
    {
        std::cout << pImage[i] << " ";
    }
    std::cout << std::endl;
    */

    #pragma omp parallel for private(ix,iCurIndex,iCompareIndex,iCurElement,iCompareElement, simValue, pCurData, pCompareData)
    for  (ix = 0; ix < xcount; ++ix)    
    {

        if (ix == 0)
        {
            #pragma omp critical
            {
                std::cout << "Within for-loop: OMP THREADS=" << omp_get_num_threads() << std::endl;
            }
        }


        for (uint32_t iy = 0; iy < ycount; ++iy)    
        {
            iCurIndex = (ix*ycount+iy) * m_iDims;
            iCurElement = ix * ycount + iy;

            pCurData = &(pImage[iCurIndex]);

            /*
            std::cout << "Current Data x=" << ix << " y=" << iy << std::endl;
            for (uint8_t ibla= 0; ibla < m_iDims; ++ibla)
            {
                std::cout << (float) pCurData[ibla] << " ";
            }
            std::cout << std::endl;
            */
            

            for (uint32_t iix = 0; iix < xcount; ++iix)    
            {
                for (uint32_t iiy = 0; iiy < ycount; ++iiy)    
                {
                    iCompareIndex = (iix*ycount+iiy) * m_iDims;
                    iCompareElement = iix * ycount + iiy;
                    pCompareData = &(pImage[iCompareIndex]);

                    if (iCurElement < iCompareElement)
                    {
                        continue;
                    }

                    simValue = this->dotProduct(pCurData, pCompareData, m_iDims, false);

                    pSim[ iCurElement * iElements + iCompareElement ] = (float) simValue;
                    pSim[ iCompareElement * iElements + iCurElement ] = (float) simValue;

                    /*
                    if ((ix == iix) && (iy == iiy))
                    {
                        if (!this->fequals(1.0f, simValue))
                        {
                            std::cerr << iCurElement << " " << iCompareElement << " " << simValue << std::endl;

                            this->dotProduct(pCurData, pCompareData, m_iDims, true);
                        }
                        assert(this->fequals(1.0f, simValue));
                    }
                    */
                }
            }
        }
    }

    /*

    for (uint32_t i = 0; i < iElements; ++i)
    {
        uint32_t coord = i * iElements + i;

        if (!fequals(pSim[coord], 1.0f))
        {
            std::cerr << i << " " << coord << " " << pSim[coord] << std::endl;
        }
    }
    */

    for (uint8_t i = 0; i < 10; ++i)
    {
        std::cout<< "i=" << pSim[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Finished processing. Created matrix with " << iNumFields << " Fields." << std::endl;


    return pSim;
    
}

bool SRM::fequals(float f1, float f2)
{
    return fabs(f1 - f2) < 0.00005 ;
}

float SRM::dotProduct(float* pData1, float* pData2, uint32_t icount, bool verbose)
{
    float outVal = 0.0f;
    float ldat1 = 0.0f;
    float ldat2 = 0.0f;

    //std::cout << "calc dot prod for elements " << icount << std::endl;

    for (uint32_t i=0; i < icount; ++i)
    {
        outVal += (float) pData1[i]* (float) pData2[i];
        ldat1 += (float) pData1[i]* (float) pData1[i];
        ldat2 += (float) pData2[i]* (float) pData2[i];
    }

    if ((ldat1 == 0.0f) && (ldat2 == 0.0f))
    {
        return 1.0f;
    }

    if (outVal == 0.0f)
    {
        return 0.0f;
    }

    ldat1 = sqrt(ldat1);
    ldat2 = sqrt(ldat2);

    float flen = ldat1 * ldat2;

    float retVal = 1.0f * ((float) outVal / (float) flen);

    if (verbose)
    {
        std::cout << outVal << " " << ldat1 << " " << ldat2 << " " << flen << " " << retVal << " " << 1.0f-retVal << " " << SRM::fequals(1.0f, retVal) << std::endl;
    }
    

    return retVal;
}

uint32_t* SRM::getSegmentedImageFloat(uint32_t xcount, uint32_t ycount, float* pImage)
{

    std::vector< PixelRegion*>* pPixelRegions = new std::vector< PixelRegion* >();
    std::vector< std::pair< PixelRegion*, PixelRegion*> *>* pEdges = new std::vector< std::pair< PixelRegion*, PixelRegion*> *>();
    std::map< int, ImageRegion*>* mapRegions = new std::map< int, ImageRegion* >();
    std::map<int, int>* pRegionsOfCardinality = new std::map<int,int>();


    uint32_t x = xcount;
    uint32_t y = ycount;

    uint32_t iImageSize = x*y;

    std::cerr << "Preparing SRM segmented Image" << std::endl;
    std::cerr << "Allocating space for segmentation" << std::endl;



    // step 1: create regions for each pixel.


    std::cerr << "Creating PixelRegions" << std::endl;

    PixelRegion **pRegions = (PixelRegion**) malloc(sizeof(PixelRegion*) * x * y);

    uint32_t iImgIndex = 0;
    uint32_t iRawIndex = 0;
    ImageRegion *pRegion;
    for (uint32_t i = 0; i < (x); ++i){
        for (uint32_t j = 0; j < (y); ++j)
        {
            iRawIndex = i*y+j;
            iImgIndex = i*y*m_iDims+j*m_iDims;
            pRegion = new ImageRegion(m_iDims);
            pRegions[iRawIndex] = new PixelRegion(iRawIndex, i, j, pRegion);
            pRegion->addPixel(pRegions[iRawIndex], &(pImage[iImgIndex]));
            //m_pRegions->insert(m_pRegions->end(), pRegion);
            pRegion->regionId = iRawIndex;
            mapRegions->insert( std::pair<int,ImageRegion*>( iRawIndex, pRegion ) );
        }
    }
    (*pRegionsOfCardinality)[1] = x*y;


    std::cerr << "Creating links between PixelRegions" << std::endl;

    for (int i = 0; i < (x-1); ++i)
        for (int j = 0; j < (y-1); ++j)
        {
            iRawIndex = i*y + j;

            pEdges->insert(pEdges->end(), new std::pair<PixelRegion*, PixelRegion*>(pRegions[iRawIndex], pRegions[iRawIndex + 1]));
            pEdges->insert(pEdges->end(), new std::pair<PixelRegion*, PixelRegion*>(pRegions[iRawIndex], pRegions[iRawIndex + y]));
        }

    for (uint32_t i = 0; i < y-1; ++i)
    {
        iRawIndex = (x-1) * y + i;
        pEdges->insert(pEdges->end(), new std::pair<PixelRegion*, PixelRegion*>(pRegions[iRawIndex], pRegions[iRawIndex + 1]));
    }

    for (uint32_t i = 0; i < x-1; ++i)
    {
        iRawIndex = i * y;
        pEdges->insert(pEdges->end(), new std::pair<PixelRegion*, PixelRegion*>(pRegions[iRawIndex], pRegions[iRawIndex + y]));
    }

    // step 2: sort regions
    std::cerr << "Sort Links" << std::endl;
    std::sort(pEdges->begin(), pEdges->end(), &(SRM::sortGradients));

    // step 3: perform merging
    // TODO what does SImage do?
    // TODO how to save output?

    uint32_t* pClusters = (uint32_t*) malloc(m_vQs.size() * sizeof(uint32_t) * x * y);

    ImageRegion *pDeleteRegion, *pMergedRegion;
    std::vector< ImageRegion* >::iterator oPRIt;
    bool test;

    uint8_t iStep=0;
    for (uint8_t iQIdx=0; iQIdx < m_vQs.size(); ++iQIdx)
    {
        float fQ = m_vQs[iQIdx];
        std::cerr << "Starting run with Q=" << fQ << std::endl;

        std::vector< std::pair<PixelRegion*, PixelRegion*>* >::iterator oIt = pEdges->begin();
        std::pair<PixelRegion*, PixelRegion*> *pCurrentEdge = NULL;

        int i = 0;
        size_t iStartRegionCount = mapRegions->size();

        for ( ; oIt != pEdges->end(); ++oIt)
        {

            pCurrentEdge = *(oIt);

            test = this->testMergeRegions(pCurrentEdge->first, pCurrentEdge->second, fQ, iImageSize, pRegionsOfCardinality);

            if (test)
            {

                pDeleteRegion = pCurrentEdge->second->getRegion();
                pMergedRegion = this->mergeRegions(pCurrentEdge->first->getRegion(), pCurrentEdge->second->getRegion(), pRegionsOfCardinality);
                mapRegions->erase( pDeleteRegion->regionId );
                //SAFEDEL(pDeleteRegion)
/*
                oPRIt = mapRegions->begin();
                for ( ; oPRIt != mapRegions->end(); ++oPRIt)
                {
                    if (*(oPRIt) == pDeleteRegion)
                    {
                        mapRegions->erase(oPRIt);
                        break;
                    }
                }
*/
            }

            ++i;

            if (i % 10000 == 0) {
                std::cerr << "Finished " << i << " of " << pEdges->size() << std::endl;
            }
        }

        std::cerr << "Finished with edges" << std::endl;

        std::cerr << "Processing Regions: " << mapRegions->size() << std::endl;

        uint32_t iClusterID = 0;
        for (  std::map<int,ImageRegion*>::iterator oIt = mapRegions->begin(); oIt != mapRegions->end(); ++oIt)
        {
            //std::cerr << "Adding Image Region " << iClusterID << std::endl;

            ImageRegion *pRegion = (*oIt).second;
            pRegion->setPixelsCluster(&(pClusters[iStep*x*y]), iClusterID);

            ++iClusterID;
        }

        ++iStep;
        size_t iEndRegionCount = mapRegions->size();

        std::cerr << "Reduced Regions from " << iStartRegionCount << " to " << iEndRegionCount << std::endl;
    }

    std::cerr << "Output clusters" << std::endl;


    return pClusters;
}


bool SRM::testMergeRegions(PixelRegion *pPR1, PixelRegion *pPR2, float fQ, uint32_t iImageSize, std::map<int, int>* pRegionsOfCardinality)
{

    ImageRegion *pR1 =pPR1->getRegion();
    ImageRegion *pR2 =pPR2->getRegion();

    if (pR1 == pR2)
        return false; //already belong to the same group -> no merge

    float fB1 = getB(pR1, fQ, iImageSize, pRegionsOfCardinality);
    float fB2 = getB(pR2, fQ, iImageSize, pRegionsOfCardinality);

    fB1 = fB1 * fB1;
    fB2 = fB2 * fB2;

    float fRegionDist = 0.0f;

    if (this->m_eSegmentMode == SEGMODE::DOTPROD)
    {
        fRegionDist = this->distanceFunctionDot(pR1, pR2);
    } else { // this->m_eSegmentMode == SEGMODE::EUCL
        fRegionDist = this->distanceFunction(pR1, pR2);
    }

    float fRegionDistSq = fRegionDist * fRegionDist;

    if (fRegionDistSq <= fB1+fB2 )
    {
        return true;
    }

    return false;
}

float SRM::distanceFunction(ImageRegion *pR1, ImageRegion *pR2)
{
    float* oAvg1 = pR1->getAvgColor();
    float* oAvg2 = pR2->getAvgColor();

    float dDist = 0.0f;
    for (uint8_t d=0; d < pR1->getDims(); ++d)
    {   
        dDist += (oAvg1[d] - oAvg2[d]) * (oAvg1[d] - oAvg2[d]);
    }

    dDist = sqrt(dDist);

    return dDist;
}

float SRM::distanceFunctionDot(ImageRegion *pR1, ImageRegion *pR2)
{
    float* oAvg1 = pR1->getAvgColor();
    float* oAvg2 = pR2->getAvgColor();

    float fDotProd = SRM::dotProduct(oAvg1, oAvg2, pR1->getDims(), false);

    return 1.0f-fDotProd;
}

bool SRM::sortGradients(std::pair<PixelRegion*, PixelRegion*> *pP1, std::pair<PixelRegion*, PixelRegion*> *pP2)
{
    float fDist1 = SRM::distanceFunction(pP1->first->getRegion(), pP1->second->getRegion());
    float fDist2 = SRM::distanceFunction(pP2->first->getRegion(), pP2->second->getRegion());

    return fDist1 < fDist2;
}

ImageRegion *SRM::mergeRegions(ImageRegion *pR1, ImageRegion *pR2, std::map<int, int>* pRegionsOfCardinality)
{  
    int size1 = pR1->size();
    int size2 = pR2->size();
    int rSize = (*pRegionsOfCardinality)[size1]-1;
    if( rSize == 0 ){
        pRegionsOfCardinality->erase(size1);
    }
    else{
        (*pRegionsOfCardinality)[size1] = rSize;
    }
    rSize = (*pRegionsOfCardinality)[size2]-1;
    if( rSize == 0 ){
        pRegionsOfCardinality->erase(size2);
    }
    else{
        (*pRegionsOfCardinality)[size2] = rSize;
    }
    (*pRegionsOfCardinality)[size1+size2] = ((*pRegionsOfCardinality)[size1+size2] + 1);

    return pR1->addRegion(pR2);
}

float SRM::getB(ImageRegion *pRegion, float fQ, uint32_t iImageSize, std::map<int, int>* pRegionsOfCardinality)
{
    float fG;
    
    if (this->m_eSegmentMode == SEGMODE::DOTPROD)
    {
        fG = 256.0f;
    } else if (this->m_eSegmentMode == SEGMODE::EUCL) {
        fG = 256.0f;
    } else {
        fG = 256.0f;
    }

    float fFac = 1.0f / (2.0f * fQ * pRegion->size());

    float fDelta = (*pRegionsOfCardinality)[pRegion->size()] * ( iImageSize * iImageSize) ;// 1 / (6 |I|^2)
    //float fDelta = this->getRegionsOfCardinality(pRegion->size()) * (6.0f * m_iImageSize * m_iImageSize) ;// 1 / (6 |I|^2)

    float fSqrt =  sqrt( fFac * log( fDelta ));

    return fG * fSqrt;
}

