# sun_viewer

The goal of this project is to create a user-friendly program for viewing and manipulating fits files, with a focus on solar physics-related applications. Qt is being used to develop the interface while the application itself is being written in python. 

In its current form there will be two files present - a .ui file created with Qt's Designer software, and a python file. Running the python file from the same location as the ui file should start the application. 

At the moment its functionality is limited to displaying images generated from user-selected fits files (scroll through with left and right arrow keys), and selecting from a few astropy stretch functions to operate on these images. 

Goals for the project:
  
  * Create a contextual slider with values dependent on the selected stretch function (specifies the "a" value in astropy's log stretch function, for       example)
  * Expand the stretch function list
  * Add noise reduction options (Savitzky-Golay, gaussians, etc)
  * Add ability to export to new fits files
  * Allow for side-by-side views of data sets to compare original and altered images
  * Incorporate coordinates from fits headers and add options for conversion between systems
  * Add tools for measuring features in images with mouse clicks
  * Add ability to create time-height plots
