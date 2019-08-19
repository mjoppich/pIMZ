
import numpy as np
from scipy import misc
import ctypes
import matplotlib.pyplot as plt
import os,sys
from collections import defaultdict
from pyimzml.ImzMLParser import ImzMLParser, browse, getionimage


baseFolder = str(os.path.dirname(os.path.realpath(__file__)))

lib = ctypes.cdll.LoadLibrary(baseFolder+'/../../cppSRM/lib/libSRM.so')

class Segmenter():

    def __init__(self):
        lib.StatisticalRegionMerging_New.argtypes = [ctypes.c_uint8, ctypes.POINTER(ctypes.c_float), ctypes.c_uint8]
        lib.StatisticalRegionMerging_New.restype = ctypes.c_void_p

        lib.SRM_processFloat.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        lib.SRM_processFloat.restype = ctypes.POINTER(ctypes.c_uint32)

    def segment_array(self, inputarray, qs=[256, 0.5, 0.25], imagedim = None):

        dims = 1

        if len(inputarray.shape) > 2:
            dims = inputarray.shape[2]

        qArr = (ctypes.c_float * len(qs))(*qs)

        self.obj = lib.StatisticalRegionMerging_New(dims, qArr, len(qs))
        
        dimage = (inputarray / np.max(inputarray)) * 255
        image_p = dimage.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        retValues = lib.SRM_processFloat(self.obj, dimage.shape[0], dimage.shape[1], image_p)

        outclust = np.ctypeslib.as_array(retValues, shape=(len(qs), dimage.shape[0], dimage.shape[1]))

        print(outclust.dtype)
        print(outclust.shape)

        outdict = {}

        for i,q in enumerate(qs):
            outdict[q] = outclust[i, :,:]

        if imagedim == None:
            imagedim = int(dims/3)

        image = inputarray[:,:,imagedim]
        image = image / np.max(image)

        return image, outdict



    def segment_image(self, imagepath, qs=[256, 0.5, 0.25]):

        #load image
        image = misc.imread(imagepath)
        image = image.astype(np.float32)
        image = image / np.max(image)

        print(image.shape)
        print(image.dtype)
        print(np.min(image), np.max(image))

        dims = 1

        if len(image.shape) > 2:
            dims = image.shape[2]

        qArr = (ctypes.c_float * len(qs))(*qs)

        self.obj = lib.StatisticalRegionMerging_New(dims, qArr, len(qs))
        
        dimage = image * 255
        image_p = dimage.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        retValues = lib.SRM_processFloat(self.obj, dimage.shape[0], dimage.shape[1], image_p)

        outclust = np.ctypeslib.as_array(retValues, shape=(len(qs), dimage.shape[0], dimage.shape[1]))

        print(outclust.dtype)
        print(outclust.shape)

        outdict = {}

        for i,q in enumerate(qs):
            outdict[q] = outclust[i, :,:]

        return image, outdict


class IMZMLExtract:

    def __init__(self, fname):
        #fname = "/mnt/d/dev/data/190724_AR_ZT1_Proteins/190724_AR_ZT1_Proteins_spectra.imzML"

        self.fname = fname
        self.parser = ImzMLParser(fname)
        self.dregions = None

        self.mzValues = self.parser.getspectrum(0)[0]

        self.find_regions()

    def get_mz_index(self, value):

        curIdxDist = 1000000
        curIdx = 0

        for idx, x in enumerate(self.mzValues):
            dist = abs(x-value)

            if dist < curIdxDist:
                curIdx = idx
                curIdxDist = dist
            
        return curIdx

    def get_region_spectra(self, regionid):

        if not regionid in self.dregions:
            return None
        
        outspectra = {}

        for coord in self.dregions[regionid]:

            spectID = self.parser.coordinates.index(coord)

            if spectID == None or spectID < 0:
                print("Invalid coordinate", coord)
                continue

            outspectra[coord] = self.parser.getspectrum( spectID )[1]

        return outspectra

    def get_region_range(self, regionid):

        allpixels = self.dregions[regionid]

        minx = min([x[0] for x in allpixels])
        maxx = max([x[0] for x in allpixels])

        miny = min([x[1] for x in allpixels])
        maxy = max([x[1] for x in allpixels])

        minz = min([x[2] for x in allpixels])
        maxz = max([x[2] for x in allpixels])

        spectraLength = 0
        for coord in self.dregions[regionid]:

            spectID = self.parser.coordinates.index(coord)

            if spectID == None or spectID < 0:
                print("Invalid coordinate", coord)
                continue

            splen = self.parser.mzLengths[spectID]

            spectraLength = max(spectraLength, splen)

        return (minx, maxx), (miny, maxy), (minz, maxz), spectraLength

    def get_region_shape(self, regionid):

        rr = self.get_region_range(regionid)
        xr,yr,zr,sc = rr

        imzeShape = [
            xr[1]-xr[0]+1,
            yr[1]-yr[0]+1
        ]

        if zr[1]-zr[0]+1 > 1:
            imzeShape.append( zr[1]-zr[0]+1 )

        imzeShape.append(sc)

        spectraShape = tuple(imzeShape)

        return spectraShape


    def get_region_array(self, regionid):

        xr,yr,zr,sc = self.get_region_range(regionid)
        rs = self.get_region_shape(regionid)
        print(rs)

        sarray = np.zeros( rs, dtype=np.float32 )

        coord2spec = self.get_region_spectra(regionid)

        for coord in coord2spec:
            xpos = coord[0]-xr[0]
            ypos = coord[1]-yr[0]

            spectra = coord2spec[coord]

            if len(spectra) < sc:
                spectra = np.pad(spectra, ((0,0),(0, sc-len(spectra) )), mode='constant', constant_values=0)

            sarray[xpos, ypos, :] = spectra

        return sarray

    def find_regions(self):

        if os.path.isfile(self.fname + ".regions"):

            print("Opening regions file for", self.fname)

            with open(self.fname + ".regions", 'r') as fin:
                self.dregions = defaultdict(list)

                for line in fin:
                    line = line.strip().split("\t")

                    coords = [int(x) for x in line]

                    self.dregions[coords[3]].append( tuple(coords[0:3]) )

            for regionid in self.dregions:

                allpixels = self.dregions[regionid]

                minx = min([x[0] for x in allpixels])
                maxx = max([x[0] for x in allpixels])

                miny = min([x[1] for x in allpixels])
                maxy = max([x[1] for x in allpixels])

                print(regionid, minx, maxx, miny, maxy)

        else:

            self.dregions = self.__detectRegions(self.parser.coordinates)

            with open(self.fname + ".regions", 'w') as outfn:

                for regionid in self.dregions:

                    for pixel in self.dregions[regionid]:

                        print("\t".join([str(x) for x in pixel]), regionid, sep="\t", file=outfn)
    
    
    def __dist(self, x,y):

        assert(len(x)==len(y))

        dist = 0
        for pidx in range(0, len(x)):

            dist += abs(x[pidx]-y[pidx])

        return dist


    def __detectRegions(self, allpixels):

        allregions = []

        for idx, pixel in enumerate(allpixels):

            if len(allregions) == 0:
                allregions.append([pixel])
                continue

            if idx % 1000 == 0:
                print("At pixel", idx , "of", len(allpixels), "with", len(allregions), "regions")


            accRegions = []

            for ridx, region in enumerate(allregions):

                #minx = min([x[0] for x in region])
                #maxx = max([x[0] for x in region])

                #miny = min([x[1] for x in region])
                #maxy = max([x[1] for x in region])

                #if pixel[0] - maxx > 100:
                #    continue

                #if pixel[0] - minx > 100:
                #    continue

                #if pixel[1] - maxy > 100:
                #    continue

                #if pixel[1] - miny > 100:
                #    continue

                for coord in region:
                    if self.__dist(coord, pixel) <= 1:
                        accRegions.append(ridx)
                        break


            if len(accRegions) == 0:
                allregions.append([pixel])

            elif len(accRegions) == 1:

                for ridx in accRegions:
                    allregions[ridx].append(pixel)

            elif len(accRegions) > 1:

                bc = len(allregions)

                totalRegion = []
                for ridx in accRegions:
                    totalRegion += allregions[ridx]

                for ridx in sorted(accRegions, reverse=True):
                    del allregions[ridx]

                allregions.append(totalRegion)

                ac = len(allregions)

                assert(ac == bc + 1 - len(accRegions))

        outregions = {}

        for i in range(0, len(allregions)):
            outregions[i] = [tuple(x) for x in allregions[i]]

        return outregions


if __name__ == '__main__':


    imze = IMZMLExtract("/mnt/d/dev/data/190724_AR_ZT1_Proteins/190724_AR_ZT1_Proteins_spectra.imzML")
    spectra = imze.get_region_array(1)

    print("Got spectra", spectra.shape)
    print("mz index", imze.get_mz_index(6662))

    seg = Segmenter()

    #image, regions = seg.segment_image("/mnt/d/dev/data/mouse_pictures/segmented/test1_smaller.png", qs=[256, 0.5, 0.25, 0.0001, 0.00001])
    image, regions = seg.segment_array(spectra, qs=[256, 0.5, 0.25, 0.0001, 0.00001, 0.000000001], imagedim=imze.get_mz_index(6662))


    f, axarr = plt.subplots(len(regions), 2)

    for i,q in enumerate(regions):

        curdata = regions[q]
        uniques = np.unique(curdata)
        print("Q", q, len(uniques))

        if len(uniques) < 100:
            print(uniques)
        print()


        axarr[i, 0].imshow( image )
        axarr[i, 1].imshow( curdata )

    plt.show()
    plt.close()