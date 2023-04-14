<html>
<head></head>
<body>
  <h1 style="color: red;">Landslide-susceptibility-LDA</h1>
</body>
</html>

This Python script generates a landslide susceptibility model using linear discriminant analysis. As input, the script requires the following variables:

- Output path
- Path of each of the thematic maps
- Path of the map with the sample inventory of morphodynamic processes to be used for training.
- Null value or no data value

## The script will generate the following:

- The script generates a text file (.txt) showing the input arguments, the results of the Shapiro-Wilks and Bartlett tests for whether or not the assumptions of the LDA method are met.
- The coefficients of the discriminant function. 
- Images in PNG format that include the covariance matrix for the classes with and without landslides, the success curve for the training and validation data, as well as a three-dimensional representation of the relationship of the training sample with each of the conditioning factors before and after applying the linear discriminant analysis.
- A Landslide Susceptibility Index (LSI) map in TIFF format.

## Installation:
To use the script, it is necessary to have Python and some additional libraries installed (numpy, os, osgeo, pandas, sklearn, math, seaborn, scipy.stats and matplotlib and the argparse module)

## Authorship:
This project was created by Nayleth Alexandra Rojas Becerra. You can contact the author at nayleth_alexandra@hotmail.com
