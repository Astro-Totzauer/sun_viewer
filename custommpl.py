from PyQt5.Qt import Qt
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, 
    NavigationToolbar2QT as NavigationToolbar)
import sys
from PyQt5 import QtGui, QtWidgets
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ContrastBiasStretch,ManualInterval,LinearStretch,MinMaxInterval,ImageNormalize, SqrtStretch,LogStretch,PowerDistStretch,PowerStretch,SinhStretch,SquaredStretch,AsinhStretch,PercentileInterval

# Connects the Qt UI file with this python file
Ui_MainWindow, QMainWindow = loadUiType('window.ui')

# This class contains all the functions that can be called by the UI. 
class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.comboBox.addItems(['Sqrt Stretch', 'Linear Stretch', 'Squared Stretch', 'Power Stretch', 'Log Stretch'])
        self.openButton.clicked.connect(self.selectFile)
        self.comboBox.activated[str].connect(self.norm_set)
        self.var_min = 0
        self.var_max = 1
        self.var_int = 1
        self.varSlider.setMinimum(self.var_min)
        self.varSlider.setMaximum(self.var_max)
        self.varSlider.setTickInterval(self.var_int)
        self.varSlider.valueChanged.connect(self.setSliderValue)
        self.i = 0
        self.a = 1
        self.norm = ImageNormalize(interval=MinMaxInterval())
        self.dates = []

# Function initiated by any key press event
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev()
        elif event.key() == Qt.Key_Right:
            self.next()
# Contextual slider behavior, for stretch functions
    def setSliderValue(self):
        self.a = self.varSlider.value()
        self.rmmpl()
        self.create_image(self.cube)
        self.addmpl(self.fig1)

# Resets the normalization parameters when the stretch option changes
# This needs work - need to add more stretch options and improve parameters for each
    def norm_set(self, text):
        stretch_dict = {'Sqrt Stretch':SqrtStretch(), 'Linear Stretch':LinearStretch(), 'Squared Stretch':SquaredStretch(), 'Power Stretch':PowerStretch(self.a), 'Log Stretch':LogStretch(self.a)}
        self.stretch_val = stretch_dict[text]
        if text == 'Log Stretch':
            self.var_max = 10000
            self.var_int = 10
        else:
            self.var_max = 10
            self.var_int = 1
        self.norm = ImageNormalize(interval=MinMaxInterval(),stretch=self.stretch_val)
        main.create_image(self.cube)
        main.rmmpl()
        main.addmpl(self.fig1)

# Adds the Qt widget necessary for displaying the matplotlib image
# mplvl is the layout object in Qt
    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()

# Removes the Qt display widget, used before a new image is displayed
    def rmmpl(self,):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()

# Displays previous image, prevents errors at boundaries of image set
    def prev(self):
        if self.i == 0:
            self.i = 0
        else:
            self.i = self.i - 1
        main.rmmpl()
        main.create_image(self.cube)
        main.addmpl(self.fig1)

# Displays next image in set. Can be used with key event or button.
    def next(self):
        if self.i == self.n - 1:
            self.i = self.n - 1
        else:
            self.i = self.i + 1
        main.rmmpl()
        main.create_image(self.cube)
        main.addmpl(self.fig1)

# Creates a 3-d ndarray from list of files and creates a
# list of fits headers
    def make_cube(self, files):
        n = len(files)
        image0 = fits.getdata(files[0])
        x = image0.shape[0]
        y = image0.shape[1]
        dataz = np.zeros((x,y,n))
        for i in range(n):
            dataz[:,:,i] = fits.getdata(files[i])
            data = fits.open(files[i])
            date_obs = data[0].header['DATE-OBS']
            self.dates.append(date_obs)
        return dataz

# Creates individual image to be displayed and generates
# accompanying date from header
    def create_image(self,cube):
        date = self.dates[self.i]
        self.image = np.flip(cube[:,:,self.i],axis=0)
        self.fig1 = Figure()
        self.axf1 = self.fig1.add_subplot(111)
        self.axf1.set_title(date)
        self.axf1.imshow(self.image, norm = self.norm, cmap='gray')

# Connected with Open button, creates interactive dialog box for user to select
# files to be given to the make_cube function
    def selectFile(self):
        home_dir = str(Path.home())
        files, filter = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Files", " ", "Fits files (*.fits)")
        files = sorted(files)
        self.cube = main.make_cube(files)
        self.n = self.cube.shape[2]
        self.image = np.flip(self.cube[:,:,0],axis=0)
        self.fig1 = Figure()
        self.axf1 = self.fig1.add_subplot(111)
        self.axf1.imshow(self.image, cmap='gray')
        main.addmpl(self.fig1)
    
# Initializes this application when python file is run
if __name__ == '__main__':

    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
