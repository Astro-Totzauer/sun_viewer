from PyQt5.Qt import Qt
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, 
    NavigationToolbar2QT as NavigationToolbar)
import sys
from PyQt5 import QtGui, QtWidgets, QtCore
import savitzkygolay
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ContrastBiasStretch,ManualInterval,LinearStretch,MinMaxInterval,ImageNormalize, SqrtStretch,LogStretch,PowerDistStretch,PowerStretch,SinhStretch,SquaredStretch,AsinhStretch,PercentileInterval,AsymmetricPercentileInterval,ZScaleInterval, BaseStretch
from astropy.wcs import WCS
from scipy.signal import savgol_filter, medfilt
# Connects the Qt UI file with this python file
Ui_MainWindow, QMainWindow = loadUiType('window.ui')

# This class contains all the functions that can be called by the UI. 
class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.comboBox.addItems(['No Stretch', 'Sqrt Stretch', 'Linear Stretch', 'Squared Stretch', 'Power Stretch', 'Log Stretch'])
        self.openButton.clicked.connect(self.selectFile)
        self.restartButton.clicked.connect(self.restart)
        self.comboBox.activated[str].connect(self.norm_set)
        self.printcoordsButton.clicked.connect(self.print_coords)
        self.var_min = 0
        self.var_max = 1
        self.var_int = 1
        self.v_min = 0
        self.v_max = 1
        self.percent = 0
        self.percent_low = 0
        self.percent_high = 100
        self.stretch_val = BaseStretch()
        self.varSlider.setMinimum(self.var_min)
        self.varSlider.setMaximum(self.var_max)
        self.varSlider.setTickInterval(self.var_int)
        self.varSlider.valueChanged.connect(self.setSliderValue)
        self.rbtn1.toggled.connect(self.noise_reduction_select)
        self.rbtn2.toggled.connect(self.noise_reduction_select)
        self.rbtn3.toggled.connect(self.noise_reduction_select)
        self.rbtn5.toggled.connect(self.interval_select)
        self.rbtn6.toggled.connect(self.interval_select)
        self.rbtn7.toggled.connect(self.interval_select)
        self.rbtn8.toggled.connect(self.interval_select)
        self.rbtn9.toggled.connect(self.interval_select)
        self.rbtn4.toggled.connect(self.cube_mean)
        self.zoomcheckBox.stateChanged.connect(self.zoom_state)
        self.undozoomButton.clicked.connect(self.undo_zoom)
        self.coordscheckBox.stateChanged.connect(self.coords_list)
        self.clearcoordsButton.clicked.connect(self.clear_coords)
        self.meantxt.returnPressed.connect(self.set_mean_window)
        self.sgwintxt.returnPressed.connect(self.set_sg_window)
        self.sgdegtxt.returnPressed.connect(self.set_sg_degree)
        self.rdiffCheck.stateChanged.connect(self.run_diff)
        self.divCheck.stateChanged.connect(self.cube_div)
        self.vmintxt.returnPressed.connect(self.set_vmin)
        self.vmaxtxt.returnPressed.connect(self.set_vmax)
        self.percenttxt.returnPressed.connect(self.set_percent)
        self.percentlowtxt.returnPressed.connect(self.set_percent_low)
        self.percenthightxt.returnPressed.connect(self.set_percent_high)
        #self.setSGWindow.returnPressed.connect(self.set_sg_window)
        #self.setSGDeg.clicked.connect(self.set_sg_degree)
        self.i = 0
        self.a = 1
        self.ix = 0
        self.iy = 0
        self.norm = None
        self.interval = None
        self.dates = []
        self.coords = []
        self.zoom_coords = []
        self.save_coords = False
        self.set_zoom_coords = False
        self.sg_window = 5
        self.sg_degree = 3
        self.fig1 = Figure()
        self.canvas = FigureCanvas(self.fig1)
        self.mplvl.addWidget(self.canvas)
        #self.axf1 = self.canvas.figure.subplots(ncols=1,nrows=1)
        self.cid = self.canvas.mpl_connect('button_press_event', self.figure_coords)

# Function initiated by any key press event
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev()
        elif event.key() == Qt.Key_Right:
            self.next()

    def mousePressEvent(self, event):
        focused_widget = QtWidgets.QApplication.focusWidget()
        if isinstance(focused_widget, QtWidgets.QLineEdit):
            focused_widget.clearFocus()
        
    def coords_list(self, checked):
        if checked:
            self.save_coords = True
        else:
            self.save_coords = False

    def print_coords(self):
        QMessageBox.about(self, 'Title',str(self.coords))

    def clear_coords(self):
        self.coords = []

    def set_percent(self):
        self.percent = float(self.percenttxt.text())
        main.percent_int()
        
    def percent_int(self):
        self.interval = PercentileInterval(self.percent)
        self.refresh_norm()
        print('Percent = ' + str(self.percent))

    def set_vmin(self):
        self.v_min = float(self.vmintxt.text())
        main.manual_int()
    
    def set_vmax(self):
        self.v_max = float(self.vmaxtxt.text())
        main.manual_int()

    def manual_int(self):
        self.interval = ManualInterval(self.v_min, self.v_max)
        main.refresh_norm()
        print('v_min, v_max = ' + str(self.v_min) + ', ' +  str(self.v_max))

    def set_percent_low(self):
        self.percent_low = float(self.percentlowtxt.text())
        main.asym_int()

    def set_percent_high(self):
        self.percent_high = float(self.percenthightxt.text())
        main.asym_int()

    def asym_int(self):
        self.interval = AsymmetricPercentileInterval(self.percent_low, self.percent_high)
        main.refresh_norm()
        print('Percentages = ' + str(self.percent_low) +', ' + str(self.percent_high))

    def refresh_norm(self):
        #if self.stretch_val == None:
            #self.norm = None
        #else:
            #self.norm = ImageNormalize(stretch=self.stretch_val, clip=True)
        self.cube = self.interval(self.cube_base)
        main.create_image(self.cube)
        print(self.interval)
        print(self.stretch_val)

# Contextual slider behavior, for stretch functions
    def setSliderValue(self):
        self.a = self.varSlider.value()
        self.rmmpl()
        self.create_image(self.cube)
        self.addmpl(self.fig1)

    def run_diff(self, state):
        if state == Qt.Checked:
            self.i = 0
            self.cube = np.diff(self.cube, axis=2)
            self.n = self.cube.shape[2]
            main.create_image(self.cube)
        else:
            self.cube = self.cube_base
            self.n = self.cube.shape[2]
            main.create_image(self.cube)

    def cube_div(self, state):
        if state == Qt.Checked:
            self.cube = self.cube_base / self.cube
            main.create_image(self.cube)
        else:
            self.cube = self.cube_base
            main.create_image(self.cube)

    def set_sg_window(self):
        self.sg_window = int(self.sgwintxt.text())
        main.savitzky_golay(self.sg_window, self.sg_degree)
        self.sgwintxt.clearFocus()

    def set_sg_degree(self):
        self.sg_degree = int(self.sgdegtxt.text())
        main.savitzky_golay(self.sg_window, self.sg_degree)
        self.sgdegtxt.clearFocus()
    
    def set_mean_window(self):
        self.mean_window = int(self.meantxt.text())
        main.cube_mean(self.mean_window)
        self.meantxt.clearFocus()

    def noise_reduction_select(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            if radioBtn.text() == 'Savitzky-Golay':
                main.savitzky_golay(self.sg_window, self.sg_degree)
            elif radioBtn.text() == 'Median':
                main.median_filter()
            elif radioBtn.text() == 'None':
                main.no_filter()

    def interval_select(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            if radioBtn.text() == 'MinMax':
                self.interval = MinMaxInterval()
                main.refresh_norm()
            elif radioBtn.text() == 'Manual':
                main.manual_int()
                main.refresh_norm()
            elif radioBtn.text() == 'Percentile':
                main.percent_int()
                main.refresh_norm()
            elif radioBtn.text() == 'AsymmetricPercentile':
                main.asym_int()
                main.refresh_norm()
            elif radioBtn.text() == 'ZScale':
                self.interval = ZScaleInterval()
                main.refresh_norm()

     

    def savitzky_golay(self, x, y):
        self.cube = savitzkygolay.filter3D(self.cube_base, self.sg_window, self.sg_degree)
        main.create_image(self.cube)

    def median_filter(self):
        self.cube = medfilt(self.cube_base)
        main.create_image(self.cube)

    def no_filter(self):
        self.cube = self.cube_base
        main.create_image(self.cube)

    def cube_mean(self, window):
        n = self.cube.shape[2]
        x = self.cube.shape[0]
        y = self.cube.shape[1]
        dataz = np.zeros((x,y,n))
        for i in range(n):
            if i < window:
                cube_1 = self.cube[:,:,0:(int(i+window))]
                window_means = np.mean(cube_1,axis=2)
                dataz[:,:,i] = window_means
            elif (n - i) < window:
                window_means = np.mean(self.cube[:,:,int(i-window):int(n)],axis=2)
                dataz[:,:,i] = window_means
            else:
                window_means = np.mean(self.cube[:,:,int(i-window):int(i+window)],axis=2)
                dataz[:,:,i] = window_means
        self.cube = dataz
        main.create_image(self.cube)

# Resets the normalization parameters when the stretch option changes
# This needs work - need to add more stretch options and improve parameters for each
    def norm_set(self, text):
        stretch_dict = {'No Stretch': None, 'Sqrt Stretch':SqrtStretch(), 'Linear Stretch':LinearStretch(), 'Squared Stretch':SquaredStretch(), 'Power Stretch':PowerStretch(self.a), 'Log Stretch':LogStretch(self.a)}
        self.stretch_val = stretch_dict[text]
        if text == 'Log Stretch':
            self.var_max = 10000
            self.var_int = 10
            self.norm = ImageNormalize(interval=self.interval, stretch=self.stretch_val, clip=True)
        elif text == 'No Stretch':
            self.norm = None
            self.stretch_val = None
        else:
            self.var_max = 10
            self.var_int = 1
            self.norm = ImageNormalize(interval=self.interval, stretch=self.stretch_val, clip=True)
        main.create_image(self.cube)

# Displays previous image, prevents errors at boundaries of image set
    def prev(self):
        if self.i == 0:
            self.i = 0
        else:
            self.i = self.i - 1
        main.create_image(self.cube)

# Displays next image in set. Can be used with key event or button.
    def next(self):
        if self.i == self.n - 1:
            self.i = self.n - 1
        else:
            self.i = self.i + 1
        main.create_image(self.cube)

    def figure_coords(self,event):
        self.ix = event.xdata
        self.iy = event.ydata
        if self.save_coords == True:
            self.coords.append([self.ix,self.iy])
        elif self.set_zoom_coords == True:
            self.zoom_coords.append([round(self.ix),round(self.iy)])
            if len(self.zoom_coords) == 2:
                main.zoom()
        self.lcdX.display(self.ix)
        self.lcdY.display(self.iy)

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

    def zoom(self):
        self.backup_cube = self.cube
        x1,x2 = self.zoom_coords[0][0], self.zoom_coords[1][0]
        y1,y2 = self.zoom_coords[0][1], self.zoom_coords[1][1]
        self.cube = self.cube[y2:y1,x1:x2,:]
        main.create_image(self.cube)
        
    def undo_zoom(self):
        self.cube = self.backup_cube
        self.zoom_coords = []
        main.create_image(self.cube)

    def zoom_state(self, checked):
        if checked:
            self.set_zoom_coords = True
        else:
            self.set_zoom_coords = False


# Creates individual image to be displayed and generates
# accompanying date from header
    def create_image(self,cube):
        date = self.dates[self.i]
        self.view_image.set_norm(self.norm)
        self.image = self.cube[:,:,self.i]
        imageMin = np.amin(self.image)
        imageMax = np.amax(self.image)
        self.view_image.set_data(self.image)
        self.axf1.set_title(date)
        self.fig1.canvas.draw_idle()
        self.lcdMin.display(imageMin)
        self.lcdMax.display(imageMax)

    def restart(self):
        self.cube = self.cube_base
        self.norm = None
        self.interval = None
        self.i = 0
        main.create_image(self.cube)

# Connected with Open button, creates interactive dialog box for user to select
# files to be given to the make_cube function
    def selectFile(self):
        home_dir = str(Path.home())
        files, filter = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Files", " ", "Fits files (*.fits)")
        self.files = sorted(files)
        hdu = fits.open(self.files[0])[0]
        self.wcs = WCS(hdu.header)
        self.axf1 = self.canvas.figure.subplots(ncols=1,nrows=1, subplot_kw={'projection': self.wcs})
        self.axf1.grid()
        self.cube = main.make_cube(self.files)
        self.cube_base = self.cube
        self.n = self.cube.shape[2]
        self.i = 0
        self.image = self.cube[:,:,0]
        self.view_image = self.axf1.imshow(self.image, cmap='gray')
        self.fig1.canvas.draw_idle()
    
# Initializes this application when python file is run
if __name__ == '__main__':

    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
