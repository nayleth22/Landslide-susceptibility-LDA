# Landslide-susceptibility-LDA

This Python script generates a landslide susceptibility model using linear discriminant analysis. As input, the script requires the following variables:

- Output path
- Path of each of the thematic maps
- Path of the map with the sample inventory of morphodynamic processes to be used for training.
- Null value or no data value

Within the 'DATA-INPUT' folder, you will find the maps used for my research work. These maps which are used for processing, are in raster format and are in the Magna-Sirgas 3116 coordinate reference system. Furthermore, they are precisely aligned, possess consistent resolution, and the classes are represented using numerical values.

Inside the 'PRETTY_DATA-INPUT' folder, you will find the thematic maps together with the landslide susceptibility map in its final presentation version.

## The script will generate the following:

- The script generates a text file (.txt) showing the input arguments, the results of the Shapiro-Wilks and Bartlett tests for whether or not the assumptions of the LDA method are met.
- The coefficients of the discriminant function. 
- Images in PNG format that include the covariance matrix for the classes with and without landslides, the success curve for the training and validation data, as well as a three-dimensional representation of the relationship of the training sample with each of the conditioning factors before and after applying the linear discriminant analysis.
- A Landslide Susceptibility Index (LSI) map in TIFF format.

## Installation:
To use the script, it is necessary to have Python and some additional libraries installed (numpy, gdal, osgeo, pandas, sklearn, math, seaborn, scipy.stats and matplotlib and the argparse module)

To install GDAL, I recommend you follow the tutorial below. https://www.linkedin.com/pulse/instalar-gdal-windows-10-yineth-castiblanco-rojas

## Authorship:
This project was created by Nayleth Alexandra Rojas Becerra. You can contact the author at nayleth_alexandra@hotmail.com

You can access the complete document of this research through the following link: https://noesis.uis.edu.co/handle/20.500.14071/14313
