from PyQt5.Qt import Qt
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import QApplication, QMessageBox, QButtonGroup
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, 
    NavigationToolbar2QT as NavigationToolbar)
import sys
import shutil
import os
import cv2
from PyQt5 import QtGui, QtWidgets, QtCore
import savitzkygolay
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ContrastBiasStretch,ManualInterval,LinearStretch,MinMaxInterval,ImageNormalize, SqrtStretch,LogStretch,PowerDistStretch,PowerStretch,SinhStretch,SquaredStretch,AsinhStretch,PercentileInterval,AsymmetricPercentileInterval,ZScaleInterval, BaseStretch
from astropy.wcs import WCS
import scipy.ndimage
from scipy import signal
from scipy.signal import savgol_filter, medfilt
from astropy.convolution import Gaussian2DKernel as astro_gaussian
from astropy.convolution import convolve as astro_convolve
from astropy.convolution import convolve_fft as astro_convolve_fft
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# Connects the Qt UI file with this python file
Ui_MainWindow, QMainWindow = loadUiType('window.ui')

# This class contains all the functions that can be called by the UI. 
class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.group = QButtonGroup()
        self.group.setExclusive(False)
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
        self.group.addButton(self.rbtn5)
        self.group.addButton(self.rbtn6)
        self.group.addButton(self.rbtn7)
        self.group.addButton(self.rbtn8)
        self.group.addButton(self.rbtn9)
        self.gridBox.stateChanged.connect(self.grid_lines)
        self.zoomcheckBox.stateChanged.connect(self.zoom_state)
        self.undozoomButton.clicked.connect(self.undo_zoom)
        self.coordscheckBox.stateChanged.connect(self.coords_list)
        self.clearcoordsButton.clicked.connect(self.clear_coords)
        self.meantxt.returnPressed.connect(self.set_mean_window)
        self.sgwintxt.returnPressed.connect(self.set_sg_window)
        self.sgdegtxt.returnPressed.connect(self.set_sg_degree)
        self.SGrunButton.clicked.connect(self.sg_run)
        self.rdiffCheck.stateChanged.connect(self.run_diff)
        self.divCheck.stateChanged.connect(self.cube_div)
        self.vmintxt.returnPressed.connect(self.set_vmin)
        self.vmaxtxt.returnPressed.connect(self.set_vmax)
        self.percenttxt.returnPressed.connect(self.set_percent)
        self.percentlowtxt.returnPressed.connect(self.set_percent_low)
        self.percenthightxt.returnPressed.connect(self.set_percent_high)
        self.msgn_k_text.returnPressed.connect(self.set_msgn_k)
        self.msgn_w_set_text.returnPressed.connect(self.new_w_set)
        self.msgn_gamma_text.returnPressed.connect(self.set_msgn_gamma)
        self.msgn_g_text.returnPressed.connect(self.set_msgn_g)
        self.msgn_h_text.returnPressed.connect(self.set_msgn_h)
        self.msgn_single_imageButton.clicked.connect(self.msgn_current_image)
        self.msgn_image_setButton.clicked.connect(self.msgn_all_images)
        self.msgn_resetButton.clicked.connect(self.msgn_reset)
        self.saveImageButton.clicked.connect(self.saveImage)
        self.saveVideoButton.clicked.connect(self.save_movie)
        #self.kernelsButton.clicked.connect(self.show_kernels)
        #self.setSGWindow.returnPressed.connect(self.set_sg_window)
        #self.setSGDeg.clicked.connect(self.set_sg_degree)
        self.msgn_k = 0.7
        self.msgn_gamma = 3.2
        self.msgn_g = 1
        self.msgn_h = 0.7
        self.msgn_k_text.setText(str(self.msgn_k))
        self.msgn_gamma_text.setText(str(self.msgn_gamma))
        self.msgn_g_text.setText(str(self.msgn_g))
        self.msgn_h_text.setText(str(self.msgn_h))
        self.w_set = []
        self.i = 0
        self.a = 1
        self.ix = 0
        self.iy = 0
        self.norm = None
        self.interval = None
        self.cube_base_on = False
        self.cube_filter_on = False
        self.cube_norm_on = False
        self.dates = []
        self.coords = []
        self.zoom_coords = []
        self.save_coords = False
        self.set_zoom_coords = False
        self.sg_window = 9
        self.sg_degree = 5
        self.sgwintxt.setText(str(self.sg_window))
        self.sgdegtxt.setText(str(self.sg_degree))
        self.kernel_width = 1
        self.kernel_std = 1
        self.kernel = 0
        self.cube_x = 0
        self.cube_y = 0
        self.cube_n = 0
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
        self.rbtn5.setChecked(False)
        self.rbtn6.setChecked(False)
        self.rbtn7.setChecked(True)
        self.rbtn8.setChecked(False)
        self.rbtn9.setChecked(False)
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
        self.rbtn5.setChecked(True)
        self.rbtn6.setChecked(False)
        self.rbtn7.setChecked(False)
        self.rbtn8.setChecked(False)
        self.rbtn9.setChecked(False)
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
        self.rbtn5.setChecked(False)
        self.rbtn6.setChecked(False)
        self.rbtn7.setChecked(False)
        self.rbtn8.setChecked(True)
        self.rbtn9.setChecked(False)
        main.refresh_norm()
        print('Percentages = ' + str(self.percent_low) +', ' + str(self.percent_high))

    def refresh_norm(self):
        #if self.stretch_val == None:
            #self.norm = None
        #else:
            #self.norm = ImageNormalize(stretch=self.stretch_val, clip=True)
        self.cube = self.interval(self.cube)
        main.create_image(self.cube)
        print(self.interval)
        print(self.stretch_val)

# Contextual slider behavior, for stretch functions
    def setSliderValue(self):
        self.a = self.varSlider.value()
        self.rmmpl()
        self.create_image(self.cube)
        self.addmpl(self.fig1)

    def grid_lines(self, state):
        if state == Qt.Checked:
            self.axf1.grid(b=True)
        else:
            self.axf1.grid(b=False)
        main.create_image(self.cube_display)

    def run_diff(self, state):
        if state == Qt.Checked:
            self.cube_nodiff = self.cube
            self.i = 0
            self.cube = np.diff(self.cube, axis=2)
            self.n = self.cube.shape[2]
            main.create_image(self.cube)
        else:
            self.cube = self.cube_nodiff
            self.n = self.cube.shape[2]
            main.create_image(self.cube)

    def cube_div(self, state):
        if state == Qt.Checked:
            main.cube_mean(5)
            self.cube = self.cube_base / (self.cube_mean1 + 0.03)
        else:
            self.cube = self.cube_base
        main.create_image(self.cube)

    def set_sg_window(self):
        self.sg_window = int(self.sgwintxt.text())
        #main.savitzky_golay(self.sg_window, self.sg_degree)
        self.sgwintxt.clearFocus()

    def set_sg_degree(self):
        self.sg_degree = int(self.sgdegtxt.text())
        #main.savitzky_golay(self.sg_window, self.sg_degree)
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
                self.rbtn5.setChecked(False)
                self.rbtn6.setChecked(True)
                self.rbtn7.setChecked(False)
                self.rbtn8.setChecked(False)
                self.rbtn9.setChecked(False)
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
                self.rbtn5.setChecked(False)
                self.rbtn6.setChecked(False)
                self.rbtn7.setChecked(False)
                self.rbtn8.setChecked(False)
                self.rbtn9.setChecked(True)
                main.refresh_norm()

# The following four functions update the relevant global variables for MSGN

    def set_msgn_k(self):
        self.msgn_k = float(self.msgn_k_text.text())
    
    def set_msgn_gamma(self):
        self.msgn_gamma = float(self.msgn_gamma_text.text())

    def set_msgn_g(self):
        self.msgn_g = float(self.msgn_g_text.text())

    def set_msgn_h(self):
        self.msgn_h = float(self.msgn_h_text.text())

    def new_w_set(self):
        self.w_set = [self.msgn_w_set_text.text()]
        self.w_set = self.w_set[0].split(",")
        print(self.w_set)

# Creates a 2D gaussian kernel for the multi-scale gaussian normalization (MSGN) functions

#    def gaussian_kernel(self, normalized=False):
#        gaussian1D = signal.gaussian(self.kernel_width, self.kernel_std)
#        gaussian2D = np.outer(gaussian1D, gaussian1D)
#        if normalized:
#            gaussian2D /= (2*np.pi*(self.kernel_std**2))
#        self.kernel = gaussian2D

    def gaussian_kernel(self):
        self.kernel = astro_gaussian(int(self.kernel_width))

# This is the majority of the steps taken in the MSGN algorithm (see Readme file for notes on this)

    def msgn_single_image(self,i):
        if self.cube_base_on == True:
            image = self.cube_base[:,:,i]
            print('base on')
        elif self.cube_norm_on == True:
            image = self.cube_norm[:,:,i]
            print('norm on')
        elif self.cube_filter_on == True:
            image = self.cube_base[:,:,i]
            print('filter on')
        main.gaussian_kernel()
        local_mean = astro_convolve_fft(image,self.kernel)
#        local_mean = scipy.ndimage.filters.convolve(image,self.kernel)
        difference = image - local_mean
        square_diff = difference**2
        local_std_image = np.sqrt(astro_convolve_fft(square_diff,self.kernel))
#        local_std_image = np.sqrt(scipy.ndimage.filters.convolve(square_diff,self.kernel))
        norm_image = (image - local_mean) / (local_std_image)
        norm_image = np.arctan(self.msgn_k*norm_image)
        return norm_image

# The last steps of the MSGN algorithm. I'll be adding lines for use on multi-image datasets soon.
    def msgn_image_set(self,i):
#        image_set = []
        #self.cube_filter = np.zeros((self.cube_x,self.cube_y,self.n))
        n_set = len(self.w_set)
        image_set = np.zeros((self.cube_x,self.cube_y,n_set))
        for j in range(n_set):
            self.kernel_width = self.w_set[j]
            print(self.kernel_width)
            image = main.msgn_single_image(i)
            image_set[:,:,j] = image


 #       for i in w_set:
 #           self.kernel_width = i
 #           print(self.kernel_width)
 #           image = main.msgn_single_image()
 #           image_set.append(image)
        if self.cube_base_on == True:
            image_0 = self.cube_base[:,:,i]
            a_0 = np.min(self.cube_base)
            a_1 = np.max(self.cube_base)
            print('Base on')
        elif self.cube_norm_on == True:
            image_0 = self.cube_norm[:,:,i]
            a_0 = np.min(self.cube_norm)
            a_1 = np.max(self.cube_norm)
            print('Norm on')
        elif self.cube_filter_on == True:
            image_0 = self.cube_base[:,:,i]
            a_0 = np.min(self.cube_base)
            a_1 = np.max(self.cube_base)
            print('Filter on')
        mean_local_norm = np.mean(image_set,axis=2)
        print(mean_local_norm.shape)
        gamma_transform = ((image_0 - a_0) / (a_1 - a_0))**(1.0 / self.msgn_gamma)
        msgn_image = (self.msgn_h * gamma_transform) + ((1 - self.msgn_h) * self.msgn_g * mean_local_norm)
        self.cube_filter[:,:,i] = msgn_image
        #self.cube_display = self.cube_filter
        #main.create_image(self.cube_display)

    def msgn_all_images(self):
        self.cube_filter = np.zeros((self.cube_x,self.cube_y,self.n))
        for i in range(self.n):
            main.msgn_image_set(i)
            print(i)
        self.cube_display = self.cube_filter
        main.create_image(self.cube_display)

    def msgn_current_image(self):
        self.cube_filter = np.zeros((self.cube_x,self.cube_y,self.n))
        main.msgn_image_set(self.i)
        self.cube_display = self.cube_filter
        main.create_image(self.cube_display)


    def msgn_reset(self):
        if self.cube_norm_on == True:
            self.cube_display = self.cube_norm
            print('Norm on')
        else:
            self.cube_display = self.cube_base
            print('Base on')
        #self.cube_display = self.cube_base2
        #self.cube_base = self.cube_base2
        main.create_image(self.cube_display)
        

    def sg_run(self):
        main.savitzky_golay(self.sg_window, self.sg_degree)

    def savitzky_golay(self, x, y):
        print(self.sg_window)
        print(self.sg_degree)
        self.cube_norm = savitzkygolay.filter3D(self.cube_base, self.sg_window, self.sg_degree)
        self.cube_display = self.cube_norm
        self.cube_norm_on = True
        self.cube_base_on = False
        self.cube_filter_on = False
        print(self.cube_norm_on)
        main.create_image(self.cube_display)

    def median_filter(self):
        self.cube = medfilt(self.cube_base)
        main.create_image(self.cube)

    def no_filter(self):
        main.create_image(self.cube_base)

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
        self.cube_mean1 = dataz

# Resets the normalization parameters when the stretch option changes
# This needs work - need to add more stretch options and improve parameters for each
    def norm_set(self, text):
        stretch_dict = {'No Stretch': None, 'Sqrt Stretch':SqrtStretch(), 'Linear Stretch':LinearStretch(), 'Squared Stretch':SquaredStretch(), 'Power Stretch':PowerStretch(self.a), 'Log Stretch':LogStretch(self.a)}
        self.stretch_val = stretch_dict[text]
        if text == 'Log Stretch':
            self.var_max = 10000
            self.var_int = 10
            self.cube = self.stretch_val(self.cube_base)
            #self.norm = ImageNormalize(interval=self.interval, stretch=self.stretch_val, clip=True)
        elif text == 'Sqrt Stretch':
            self.cube = np.sqrt(self.cube)
        elif text == 'No Stretch':
            self.norm = None
            self.cube = self.cube_base
        else:
            self.var_max = 10
            self.var_int = 1
            self.cube = self.stretch_val(self.cube_base, clip=False)
            #self.norm = ImageNormalize(interval=self.interval, stretch=self.stretch_val, clip=True)
        main.create_image(self.cube)

# Displays previous image, prevents errors at boundaries of image set
    def prev(self):
        if self.i == 0:
            self.i = 0
        else:
            self.i = self.i - 1
        main.create_image(self.cube_display)

# Displays next image in set. Can be used with key event or button.
    def next(self):
        if self.i == self.n - 1:
            self.i = self.n - 1
        else:
            self.i = self.i + 1
        main.create_image(self.cube_display)

    def figure_coords(self,event):
        self.ix = event.xdata
        self.iy = event.ydata
        if self.save_coords == True:
            self.coords.append([self.ix,self.iy])
        elif self.set_zoom_coords == True:
            self.zoom_coords.append([round(self.ix),round(self.iy)])
            if len(self.zoom_coords) == 2:
                main.zoom()
        pix_value = self.cube_display[round(self.iy),round(self.ix),self.i]
        self.lcdVal.display(pix_value)
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
        self.backup_cube = self.cube_display
        self.hdu_backup = self.hdu
        self.wcs_backup = self.wcs
        x1,x2 = self.zoom_coords[0][0], self.zoom_coords[1][0]
        y1,y2 = self.zoom_coords[0][1], self.zoom_coords[1][1]
        self.cube_display = self.cube_display[y2:y1,x1:x2,:]
        #pos_x = abs(x2-x1)/2.0
        #pos_y = abs(y2-y1)/2.0
        #cutout = Cutout2D(self.cube, position = (pos_x,pos_y), size = (abs(x2-x1),abs(y2-y1)), wcs=self.wcs)
        #self.hdu.header.update(cutout.wcs.to_header())
        #self.wcs = WCS(self.hdu.header)
        #self.axf1 = self.canvas.figure.subplots(ncols=1,nrows=1, subplot_kw={'projection': self.wcs})
        main.create_image(self.cube_display)

    def undo_zoom(self):
        self.cube_display = self.backup_cube
        self.zoom_coords = []
        #self.hdu = self.hdu_backup
        #self.wcs = self.wcs_backup
        #self.axf1 = self.canvas.figure.subplots(ncols=1,nrows=1, subplot_kw={'projection': self.wcs})
        main.create_image(self.cube_display)

    def zoom_state(self, checked):
        if checked:
            self.set_zoom_coords = True
        else:
            self.set_zoom_coords = False

    def save_movie(self):
        cwd = os.getcwd()
        images_folder = os.path.join(cwd,'movie_temp')
        os.makedirs(images_folder)
        for i in range(self.n):
            date = self.dates[i]
            self.view_image.set_norm(self.norm)
            self.image = self.cube_display[:,:,i]
            imageMin = np.amin(self.image)
            imageMax = np.amax(self.image)
            self.view_image.set_data(self.image)
            self.axf1.set_title(date)
            self.fig1.canvas.draw_idle()
            self.lcdMin.display(imageMin)
            self.lcdMax.display(imageMax)
            image_name = 'image'+str(i)+'.png'
            file_name = os.path.join(images_folder,image_name)
            self.fig1.canvas.figure.savefig(file_name)
        file_path = QFileDialog.getSaveFileName(self,"Save Video","","Video (*.mp4)")
        video_name = file_path[0].rsplit('/',1)[1]
        images = [img for img in os.listdir(images_folder) if img.endswith (".png")]
        frame = cv2.imread(os.path.join(images_folder, images[0]))
        height,width,layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 1, (width,height))
        for image in images:
            video.write(cv2.imread(os.path.join(images_folder, image)))
        cv2.destroyAllWindows()
        video.release()




# Creates individual image to be displayed and generates
# accompanying date from header
    def create_image(self,cube):
        date = self.dates[self.i]
        self.view_image.set_norm(self.norm)
        self.image = cube[:,:,self.i]
        imageMin = np.amin(self.image)
        imageMax = np.amax(self.image)
        self.view_image.set_data(self.image)
        self.axf1.set_title(date)
        self.fig1.canvas.draw_idle()
        self.lcdMin.display(imageMin)
        self.lcdMax.display(imageMax)

    def restart(self):
        self.cube_display = self.cube_base
        self.norm = None
        self.interval = None
        self.i = 0
        self.rbtn5.setChecked(False)
        self.rbtn6.setChecked(False)
        self.rbtn7.setChecked(False)
        self.rbtn8.setChecked(False)
        self.rbtn9.setChecked(False)
        self.rdiffCheck.setChecked(False)
        self.comboBox.setCurrentText('No Stretch')
        self.percenttxt.clear()
        self.vmintxt.clear()
        self.vmaxtxt.clear()
        self.percentlowtxt.clear()
        self.percenthightxt.clear()
        main.create_image(self.cube_display)

    def saveImage(self):
        file_path = QFileDialog.getSaveFileName(self,"Save Image","","Images (*.png *jpg *.tiff)")
        if file_path == "":
            return
        print(file_path)
        self.fig1.canvas.figure.savefig(file_path[0])

# Connected with Open button, creates interactive dialog box for user to select
# files to be given to the make_cube function
    def selectFile(self):
        home_dir = str(Path.home())
        files, filter = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Files", " ", "Fits files (*.fits)")
        self.files = sorted(files)
        self.hdu = fits.open(self.files[0])[0]
        self.wcs = WCS(self.hdu.header)
        self.axf1 = self.canvas.figure.subplots(ncols=1,nrows=1, subplot_kw={'projection': self.wcs})
        self.cube_base = main.make_cube(self.files)
        self.cube_display = self.cube_base
        self.cube_base2 = main.make_cube(self.files)
        self.cube_x = self.cube_base.shape[0]
        self.cube_y = self.cube_base.shape[1]
        self.cube_n = self.cube_base.shape[2]
#        self.cube = self.cube[725:825,950:1280,:]
        self.n = self.cube_base.shape[2]
        self.i = 0
        self.cube_base_on = True
        self.image = self.cube_display[:,:,0]
        self.view_image = self.axf1.imshow(self.image, cmap='gray')
        self.fig1.canvas.draw_idle()
    
# Initializes this application when python file is run
if __name__ == '__main__':

    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
