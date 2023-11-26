import gdal
import gdalconst
import numpy as np
import prism

class BilFile(object):

    def __init__(self, bil_file):
        self.bil_file = bil_file
        self.hdr_file = bil_file.split('.')[0]+'.hdr'

    def get_array(self, mask=None):
        self.nodatavalue, self.data = None, None
        gdal.GetDriverByName('EHdr').Register()
        img = gdal.Open(self.bil_file, gdalconst.GA_ReadOnly)
        band = img.GetRasterBand(1)
        self.nodatavalue = band.GetNoDataValue()
        self.ncol = img.RasterXSize
        self.nrow = img.RasterYSize
        geotransform = img.GetGeoTransform()
        self.originX = geotransform[0]
        self.originY = geotransform[3]
        self.pixelWidth = geotransform[1]
        self.pixelHeight = geotransform[5]
        self.data = band.ReadAsArray()
        self.data = np.ma.masked_where(self.data==self.nodatavalue, self.data)
        if mask is not None:
            self.data = np.ma.masked_where(mask==True, self.data)
        return self.nodatavalue, self.data

def getPrecipData(years=None):
    grid_pnts = prism.getGridPointsFromTxt()
    flrd_pnts = np.array(pd.read_csv(r'D:\truncated\PrismGridPointsFlrd.csv').grid_code)
    mask = prism.makeGridMask(grid_pnts, grid_codes=flrd_pnts)
    for year in years:
        bil = r'/vsizip/G:\truncated\PRISM_ppt_stable_4kmM2_{0}_all_bil.zip\PRISM_ppt_stable_4kmM2_{0}_bil.bil'.format(year)
        b = prism.BilFile(bil)
        nodatavalue, data = b.get_array(mask=mask)
        data *= mm_to_in
        b.write_to_csv(data, 'PrismPrecip_{}.txt'.format(year))
    return

# Get datasets
years = range(1950, 2011, 5)
getPrecipData(years=years)