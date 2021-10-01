# Preprocessing Pipeline 🪠

## Stage 0 (not tracked)

### Data origin
Data is sources from the [The luxembourgish data platform](https://data.public.lu). 

- 2017: [RGB Image](https://s3.eu-central-1.amazonaws.com/download.data.public.lu/resources/orthophoto-officelle-du-grand-duche-de-luxembourg-edition-2017/20180914-084015/ortho2017_RVB_pays.jp2), [NIR Image](https://s3.eu-central-1.amazonaws.com/download.data.public.lu/resources/orthophoto-officielle-du-grand-duche-de-luxembourg-edition-2017/20210720-085210/LUX17_CIR_v3.jp2)
- 2018: [RGB Image](https://s3.eu-central-1.amazonaws.com/download.data.public.lu/resources/orthophoto-officelle-du-grand-duche-de-luxembourg-edition-2018/ortho2018_RGB_pays.jp2), [NIR Image](https://s3.eu-central-1.amazonaws.com/download.data.public.lu/resources/orthophoto-officelle-du-grand-duche-de-luxembourg-edition-2018/ortho2018_CIR_pays.jp2)
- 2019: [RGB Image](https://s3.eu-central-1.amazonaws.com/download.data.public.lu/resources/orthophoto-officelle-du-grand-duche-de-luxembourg-edition-2019/20191202-111633/ortho_2019.jp2), [NIR Image](https://s3.eu-central-1.amazonaws.com/download.data.public.lu/resources/orthophoto-officelle-du-grand-duche-de-luxembourg-edition-2019/20191202-111633/ortho_2019ir.jp2)

Data is provided as JPEG2000 encoded 3-band images with 20cm resolution and projected in EPSG:2169 (LUREF).  

### Initial recombination and reprojection

Prior to the DVC-tracked preprocessing steps the raw data was reorganized as follows (the final output files are tracked in this repo though):

```bash
# year 2017 data
YEAR=2017
RGB=ortho2017_RVB_pays.jp2
NIR=LUX17_CIR_v3.jp2

# year 2018 data
YEAR=2018
RGB=ortho2018_RGB_pays.jp2
NIR=ortho2018_CIR_pays.jp2

# year 2019 data
#YEAR=2019
#RGB=ortho_2019.jp2
#NIR=ortho_2019ir.jp2

# merge bands (allow projection difference since one source has a differnt projection string but correct projection parameters)
gdalbuildvrt b1.vrt -b 1 ${RGB} 
gdalbuildvrt b2.vrt -b 2 ${RGB}
gdalbuildvrt b3.vrt -b 3 ${RGB}
gdalbuildvrt b4.vrt -b 1 ${NIR}
gdalbuildvrt -allow_projection_difference -separate allbands.vrt b1.vrt b2.vrt b3.vrt b4.vrt

gdal_translate -of GTiff -co "COMPRESS=LZW" -co "PREDICTOR=2" -co "TILED=YES" -co "BIGTIFF=YES" -co "ALPHA=NO" -colorinterp_1 "red" -colorinterp_2 "green" -colorinterp_3 "blue" allbands.vrt ortho_ms_${YEAR}.tif

# reproject and limit extent to common area
gdalwarp -co "NUM_THREADS=ALL_CPUS" -multi  -overwrite -ot Byte -ts 302346 420442 -te 263584.672 5479988.018 324060.605 5564085.859 -t_srs '+proj=utm +zone=32 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs' -r average ortho_ms_2017.tif ortho_ms_${YEAR}_EPSG3044.vrt
gdal_translate -co "NUM_THREADS=ALL_CPUS" -co "COMPRESS=LZW" -co "PREDICTOR=2" -co "TILED=YES" -co "BIGTIFF=YES" -co "ALPHA=NO" ortho_ms_${YEAR}_EPSG3044.vrt ortho_ms_${YEAR}_EPSG3044.tif

```
