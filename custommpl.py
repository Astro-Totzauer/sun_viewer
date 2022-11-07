from PyQt5.Qt import Qt
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import QApplication, QMessageBox, QButtonGroup
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from reproject import reproject_interp, reproject_exact
from reproject import reproject_adaptive
from sunpy.coordinates import Helioprojective
import sunpy.map
from sunpy.image.coalignment import mapsequence_coalign_by_match_template as mc_coalign
import astropy.units as u
import warnings
from sunpy.visualization import axis_labels_from_ctype, wcsaxes_compat
from matplotlib.figure import Figure 
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, 
    NavigationToolbar2QT as NavigationToolbar)
import sys
from natsort import natsorted
import shutil
from astropy import *
import sunpy.map
import tqdm
import sunkit_image.enhance as enhance
import os
from datetime import datetime
import h5py
import moviepy.video.io.ImageSequenceClip
import cv2
from PyQt5 import QtGui, QtWidgets, QtCore
import savitzkygolay
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from astropy.nddata import Cutout2D
from astropy.visualization import ContrastBiasStretch,ManualInterval,LinearStretch,MinMaxInterval,ImageNormalize, SqrtStretch,LogStretch,PowerDistStretch,PowerStretch,SinhStretch,SquaredStretch,AsinhStretch,PercentileInterval,AsymmetricPercentileInterval,ZScaleInterval, BaseStretch
from astropy.wcs import WCS
import scipy.ndimage
from scipy.spatial import distance
from scipy.interpolate import interp2d
from scipy import signal
from pathlib import Path
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
        #self.ng_rbtn.toggled.connect(self.noise_reduction_select)
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
        #self.ng_cube_size_x_txt.returnPressed.connect(self.set_ng_cube_size_x)
        #self.ng_cube_size_y_txt.returnPressed.connect(self.set_ng_cube_size_y)
        #self.ng_cube_size_t_txt.returnPressed.connect(self.set_ng_cube_size_t)
        #self.ng_cube_div_txt.returnPressed.connect(self.set_ng_cube_div)
        self.SGrunButton.clicked.connect(self.sg_run)
        self.rdiffCheck.stateChanged.connect(self.run_diff)
        #self.ng_run_btn.clicked.connect(self.noisegate)
        self.divCheck.stateChanged.connect(self.cube_div)
        self.expnormCheck.stateChanged.connect(self.exp_norm)
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
        self.filesaveButton.clicked.connect(self.save_files)
        self.sunkit_mgnButton.clicked.connect(self.sunkit_mgn)
        self.sessionButton.clicked.connect(self.save_session)
        self.cmapSelect.activated.connect(self.cmap_change)
        self.importButton.clicked.connect(self.import_session)
        self.files1Button.clicked.connect(self.mosaic_load_files_1)
        self.files2Button.clicked.connect(self.mosaic_load_files_2)
        self.mosaicButton.clicked.connect(self.mosaic_files)
        self.matchButton.clicked.connect(self.match_filesets)
        self.thplotButton.clicked.connect(self.t_h_start)
        #self.thradButton.clicked.connect(self.t_h_radial)
        self.mosaicfileButton.clicked.connect(self.mosaic_from_file)
        #self.kernelsButton.clicked.connect(self.show_kernels)
        #self.setSGWindow.returnPressed.connect(self.set_sg_window)
        #self.setSGDeg.clicked.connect(self.set_sg_degree)
        self.radialBox.stateChanged.connect(self.radial_coords)
        self.msgn_k = 0.7
        self.msgn_gamma = 3.2
        self.msgn_g = 1
        self.msgn_h = 0.7
        self.x_axis1 = []
        self.x_axis2 = []
        self.y_axis1 = []
        self.y_axis2 = []
        self.msgn_k_text.setText(str(self.msgn_k))
        self.msgn_gamma_text.setText(str(self.msgn_gamma))
        self.msgn_g_text.setText(str(self.msgn_g))
        self.msgn_h_text.setText(str(self.msgn_h))
        self.w_set = []
        self.i = 0
        self.a = 1
        self.lines = None
        #self.t_h_lines = None
        self.ix = 0
        self.iy = 0
        self.x0 = None 
        self.y0 = None 
        self.norm = None
        self.interval = None
        self.cube_base_on = False
        self.cube_filter_on = False
        self.cube_norm_on = False
        self.mosaic_cube_on = False
        self.dates = []
        self.coords = []
        self.ng_cube_size_x = None
        self.ng_cube_size_y = None
        self.ng_cube_size_t = None
        self.ng_cube_div = None
        self.t_h_dates = []
        self.zoom_coords = []
        self.save_coords = False
        self.set_zoom_coords = False
        self.sg_window = 9
        self.sg_degree = 5
        self.t_h_select_point = False
        self.t_h_roll = False
        self.sgwintxt.setText(str(self.sg_window))
        self.sgdegtxt.setText(str(self.sg_degree))
        self.kernel_width = 1
        self.kernel_std = 1
        self.kernel = 0
        self.cube_x = 0
        self.cube_y = 0
        self.cube_n = 0
        self.mosaic_files_list_1 = []
        self.mosaic_files_list_2 = []
        self.mosaic_primary_list = []
        self.mosaic_secondary_list = []
        self.images_folder = None
        self.mosaic_cube = None
        self.cmap_index = 3
        self.t_h_i = 0
        self.file_list = []
        self.plot_coords = []
        self.plot_image = None
        self.view_coords_image = None
        self.view_plot = None
        self.coords_x_vals = []
        self.coords_y_vals = []
        self.coords_pix_val = None
        self.coords_n = None
        self.coords_x0 = 0
        self.coords_x1 = 0
        self.coords_y0 = 0
        self.coords_y1 = 0
        self.r_clicked = False
        self.sheet_cube = None
        self.coords_n_y = None
        self.coords_fit = None
        self.coords_points = None
        self.coords_y = 0
        self.coords_z = None
        self.coords_f = None
        self.r_coords_x_vals = []
        self.r_coords_y_vals = []
        self.coords_x_start = None
        self.coords_x_end = None
        self.coords_x_new = None
        self.coords_y_new = None
        self.coords_i = 0
        self.line1 = None
        self.line2 = None
        self.t_h_line1 = None
        self.t_h_line2 = None
        self.plot_cube = None
        self.t_h_crop_array = None
        self.t_h_crop_i = 0
        self.coords_fig, self.coords_ax = plt.subplots()
        self.t_h_rad_fig, self.t_h_rad_ax = plt.subplots()
        self.t_h_fig, self.t_h_ax = plt.subplots()
        self.t_h_plot_fig, self.t_h_plot_ax = plt.subplots()
        self.t_h_crop_fig, self.t_h_crop_ax = plt.subplots()
        self.compare_sets_fig, self.compare_sets_ax = plt.subplots(2,1)
        self.mosaic_files_list_1_primary = None
        self.t_h_array = None
        self.mosaic_final_list_1 = []
        self.mosaic_final_list_2 = []
        self.mosaic_files_list_1 = []
        self.mosaic_files_list_2 = []
        self.t_h_cube = None
        self.current_cmap='gray'
        self.fig1 = Figure()
        self.view_t_h_image = None
        self.view_t_h_plot = None
        self.view_t_h_crop = None
        self.h5file1 = None
        self.h5file2 = None
        self.h5cube1 = None
        self.h5cube2 = None
        self.canvas = FigureCanvas(self.fig1)
        self.mplvl.addWidget(self.canvas)
        #self.axf1 = self.nvas.figure.subplots(ncols=1,nrows=1)ca
        #self.cid = self.canvas.mpl_connect('button_press_event', self.figure_coords)
        self.cid = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid2 = self.t_h_fig.canvas.mpl_connect('key_press_event', self.keyPressEvent2)
        self.cid3 = self.t_h_fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid4 = self.t_h_crop_fig.canvas.mpl_connect('key_press_event', self.keyPressEvent2)
        self.cid5 = self.t_h_crop_fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid6 = self.compare_sets_fig.canvas.mpl_connect('key_press_event', self.keyPressEvent3)
        self.cmaps = {0:None, 1:'goes-rsuvi131', 2:'goes-rsuvi195', 3:'gray'}

# Function initiated by any key press event
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev()
        elif event.key() == Qt.Key_Right:
            self.next()

    def keyPressEvent2(self, event):
        if event.key == 'p':
            self.show_coords()
        elif event.key == 't':
            self.show_plot()
        elif event.key == 'left':
            self.coords_prev()
            self.t_h_crop_coords_prev()
        elif event.key == 'right':
            self.coords_next()
            self.t_h_crop_coords_next()

    def keyPressEvent3(self, event):
        if event.key == 'left':
            self.compare_sets_coords_prev()
        elif event.key == 'right':
            self.compare_sets_coords_next()

    def mosaic_load_files_1(self):
        home_dir = str(Path.home())
        files, filter = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Files", " ", "Fits files (*.fts)")
        self.mosaic_files_list_1 = sorted(files)
        self.hdu = fits.open(self.mosaic_files_list_1[0])[0]
        self.wcs = WCS(self.hdu.header)
        n = len(self.mosaic_files_list_1)
        for i in range(n):
            fits_data = fits.open(self.mosaic_files_list_1[i])
            #date_obs = fits_data[0].header['DATE-OBS']
            #self.dates.append(date_obs)
        self.axf1 = self.canvas.figure.subplots(ncols=1,nrows=1, subplot_kw={'projection': self.wcs})

    def mosaic_load_files_2(self):
        home_dir = str(Path.home())
        files, filter = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Files", " ", "Fits files (*.fits)")
        self.mosaic_files_list_2 = sorted(files)
        n = len(self.mosaic_files_list_2)
        for i in range(n):
            fits_data = fits.open(self.mosaic_files_list_2[i])
            date_obs = fits_data[0].header['DATE-OBS']
            self.dates.append(date_obs)

    def select_file_1(self):
        file_name1 = QFileDialog.getOpenFileName(self,'Open Session File')
        session_file1 = h5py.File(file_name[0],'r')
        self.h5cube1 = session_file1['cube_display']

    def select_file_2(self):
        file_name2 = QFileDialog.getOpenFileName(self,'Open Session File')
        session_file2 = h5py.File(file_name[0],'r')
        self.h5cube2 = session_file1['cube_display']




    def match_filesets(self):
        self.mosaic_final_list_1 = []
        self.mosaic_final_list_2 = []
        files_1 = self.mosaic_files_list_1
        files_2 = self.mosaic_files_list_2
        files_1_hdu1 = fits.open(files_1[0])
        files_1_hdu2 = fits.open(files_1[1])
        files_2_hdu1 = fits.open(files_2[0])
        files_2_hdu2 = fits.open(files_2[1])
        files_1_t1 = files_1_hdu1[0].header['DATE-OBS']
        files_1_t2 = files_1_hdu2[0].header['DATE-OBS']
        files_2_t1 = files_2_hdu1[0].header['DATE-OBS']
        files_2_t2 = files_2_hdu2[0].header['DATE-OBS']
        files_1_t1 = Time(files_1_t1,format='isot')
        files_1_t2 = Time(files_1_t2,format='isot')
        files_2_t1 = Time(files_2_t1,format='isot')
        files_2_t2 = Time(files_2_t2,format='isot')
        files_1_cadence = files_1_t2 - files_1_t1
        files_2_cadence = files_2_t2 - files_2_t1
        #if files_1_cadence < files_2_cadence:
        #    primary_files = files_1
        #    primary_cadence = files_1_cadence
        #    secondary_files = files_2
        #    secondary_cadence = files_2_cadence
        #else: 
        #    primary_files = files_2
        #    primary_cadence = files_2_cadence
        #    secondary_files = files_1
        #    secondary_cadence = files_1_cadence
        primary_files = files_2
        primary_cadence = files_2_cadence
        secondary_files = files_1
        secondary_cadence = files_1_cadence
        for i in range(len(primary_files)):
            hdu1 = fits.open(primary_files[i])
            t1 = hdu1[0].header['DATE-OBS']
            t1 = Time(t1,format='isot')
            t_deltas = np.array([])
            print(t1)
            for j in range(len(secondary_files)):
                hdu2 = fits.open(secondary_files[j])
                t2 = hdu2[0].header['DATE-OBS']
                t2 = Time(t2,format='isot')
                t_delta = t1 - t2
                t_delta = np.abs(t_delta)
                t_deltas = np.append(t_deltas,t_delta.value)
            t_delta_min_index = np.where(t_deltas == np.amin(t_deltas))
            t_delta_min_index = t_delta_min_index[0]
            t_delta_min_index = int(t_delta_min_index)
            self.mosaic_final_list_2.append(primary_files[i])
            self.mosaic_final_list_1.append(secondary_files[t_delta_min_index])
            print('SUVI = ' + str(primary_files[i]))
            print('KCOR = ' + str(secondary_files[t_delta_min_index]))
        n_files = len(self.mosaic_final_list_1)
        print(str(n_files) + ' files were matched.')

    def mosaic_files(self):
        self.hdu = fits.open(self.mosaic_final_list_1[0])[0]
        self.wcs = WCS(self.hdu.header)
        n = len(self.mosaic_final_list_1)
        for i in range(n):
            fits_data = fits.open(self.mosaic_final_list_1[i])
            date_obs = fits_data[0].header['DATE-OBS']
            self.dates.append(date_obs)
        self.axf1 = self.canvas.figure.subplots(ncols=1,nrows=1, subplot_kw={'projection': self.wcs})
        sequence = []
        results = []
        outputs = []
        kcor_map_dims = sunpy.map.Map(self.mosaic_final_list_1[0])
        #n_x = kcor_map_dims.data.shape[0]
        #n_y = kcor_map_dims.data.shape[1]
        n_x = 1024
        n_y = 1024
        radius = 275
        center = (int(n_x/2), int(n_y/2))
        y,x = np.ogrid[:n_y, :n_x]
        dist_from_center = np.sqrt((x-center[0])**2 + (y-center[1])**2)
        mask = dist_from_center <= radius

        for fn_2, fn_1 in zip(self.mosaic_final_list_2, self.mosaic_final_list_1):
            hdu_2 = fits.open(fn_2)
            rot_angle = hdu_2[0].header['CROTA']
            rot_angle += 0.98
            hdu_2[0].header['PC1_1'] = np.cos(rot_angle*np.pi/180.0)
            hdu_2[0].header['PC1_2'] = -np.sin(rot_angle*np.pi/180.0)
            hdu_2[0].header['PC2_1'] = np.sin(rot_angle*np.pi/180.0)
            hdu_2[0].header['PC2_2'] = np.cos(rot_angle*np.pi/180.0)
            hdu_2[0].header['CROTA'] = rot_angle
            del hdu_2[0].header['OBSGEO-X']
            del hdu_2[0].header['OBSGEO-Y']
            del hdu_2[0].header['OBSGEO-Z']
    
            hdu_2[0].data *= (1.0/hdu_2[0].data.max())
            #kcor_hdu = fits.open(kcor_fn)
            #kcor_hdu[0].data *= (kcor_hdu[0].data/kcor_hdu[0].data.max())
    
            map_2 = sunpy.map.Map(hdu_2[0].data,hdu_2[0].header)
            map_1 = sunpy.map.Map(fn_1)
            #kcor_map = sunpy.map.Map(kcor_hdu[0].data,kcor_hdu[0].header)
    
            hdu = fits.open(fn_1)
            header = hdu[0].header

            with Helioprojective.assume_spherical_screen(map_2.observer_coordinate, only_off_disk=True):
                output, footprint = reproject_adaptive(map_2, map_1.wcs, map_1.data.shape)

            result = map_1.data.copy()
            result *= (1.0/result.max())
            results.append(result)
            outputs.append(output)

        for result, output in zip(results, outputs):
            result[mask] = output[mask]

            map_temp = sunpy.map.Map(result,header)
            sequence.append(map_temp)

        new_seq = sunpy.map.Map(sequence, sequence=True)
        new_seq_data = new_seq.as_array()

        self.cube_base = new_seq_data
        self.cube_display = self.cube_base
        self.backup_cube = self.cube_display
        self.mosaic_cube = self.cube_display
        self.mosaic_cube_on = True
        self.cube_x = self.cube_base.shape[0]
        self.cube_y = self.cube_base.shape[1]
        self.cube_n = self.cube_base.shape[2]
#        self.cube = self.cube[725:825,950:1280,:]
        self.n = self.cube_base.shape[2]
        self.i = 0
        self.cube_base_on = True
        self.image = self.cube_display[:,:,0]
        self.view_image = self.axf1.imshow(self.image)
        self.fig1.canvas.draw_idle()

    def mosaic_from_file(self):
        self.mosaic_cube_on = True
        self.cube_x = self.cube_base.shape[0]
        self.cube_y = self.cube_base.shape[1]
        self.cube_n = self.cube_base.shape[2]
        self.n = self.cube_base.shape[2]
        self.i = 0
        self.cube_base_on = True
        self.cube_display = np.flip(self.cube_display,axis=0)
        self.cube_display = np.flip(self.cube_display,axis=1)
        self.image = self.cube_display[:,:,0]
        self.view_image = self.axf1.imshow(self.image,cmap=self.current_cmap)
        self.fig1.canvas.draw_idle()

    def radial_coords(self,state):
        if state == Qt.Checked:
            self.r_coords_x_vals = []
            self.r_coords_y_vals = []
            self.r_clicked = True
            print(self.r_clicked)
        else:
            self.r_clicked = False
            if len(self.lines) == 2:
                self.axf1.lines[1].remove()
                self.axf1.lines[0].remove()
                self.lines = None
                self.coords_x_vals = []
                self.coords_y_vals = []
                self.fig1.canvas.draw()

    def coords_prev(self):
        if self.t_h_select_point == True:
            if self.t_h_i == 0:
                self.t_h_i == 0
            else:
                self.t_h_i = self.t_h_i -1
        else:
            if self.i == 0:
                self.i = 0
            else:
                self.i = self.i - 1
        main.create_image(self.cube_display)
        main.create_t_h_image(self.t_h_cube)

    def coords_next(self):
        if self.t_h_select_point == True:
            if self.t_h_i == self.cube_n:
                self.t_h_i = self.t_h_i
            else:
                self.t_h_i = self.t_h_i + 1

        else:
            if self.i == self.cube_n - 1:
                self.i = self.cube_n - 1
            else:
                self.i = self.i + 1
                self.t_h_i = self.t_h_i + 1
        main.create_image(self.cube_display)
        main.create_t_h_image(self.t_h_cube)


    def t_h_crop_coords_prev(self):
        if self.t_h_roll == True:
            if self.t_h_crop_i == 0:
                self.t_h_crop_i == 0
            else:
                self.t_h_crop_i = self.t_h_crop_i - 1
        else:
            pass
        main.create_t_h_crop_image(self.t_h_crop_array)
        #main.t_h_indicator()

    def t_h_crop_coords_next(self):
        if self.t_h_roll == True:
            if self.t_h_crop_i == self.cube_n:
                self.t_h_crop_i = self.t_h_crop_i
            else:
                self.t_h_crop_i = self.t_h_crop_i + 1
        else:
            pass
        main.create_t_h_crop_image(self.t_h_crop_array)
        #main.t_h_indicator()

    def coords_create_image(self):
        self.coords_ax.set_title(self.coords_i)
        self.plot_image = self.plot_cube[:,:,self.coords_i]
        self.view_coords_image.set_data(self.plot_image)
        self.coords_fig.canvas.draw_idle()

    def t_h_start(self):
        self.t_h_cube = np.zeros((int(self.cube_display.shape[0]/2),int(self.cube_display.shape[1]),int(self.cube_display.shape[2])))
        #self.cube_display = self.t_h_div(self.cube_display)
        for i in range(self.cube_display.shape[2]):
            image = self.cube_display[:,:,i]
            polar_img = cv2.warpPolar(image, (int(self.cube_display.shape[0]/2),int(self.cube_display.shape[1])), (int(image.shape[0]/2),int(image.shape[1]/2)+30), int(image.shape[0]/2)-30, cv2.WARP_POLAR_LINEAR)
            polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_CLOCKWISE)
            a = np.array(polar_img)
            self.t_h_cube[:,:,i] = a
        self.t_h_cube = np.flip(self.t_h_cube,axis=0)
        self.t_h_fig.show()
        self.view_t_h_image = self.t_h_ax.imshow(self.t_h_cube[:,:,self.i])
        main.create_t_h_image(self.t_h_cube)
        self.t_h_select_point = True
    
    def t_h_crop(self):
        x0 = int(self.r_coords_x_vals[0])
        x1 = int(self.r_coords_x_vals[1])
        x = abs(x1-x0)
        self.t_h_crop_array = self.t_h_cube[:,x0:x1,:]
        self.t_h_crop_fig.show()
        self.view_t_h_crop = self.t_h_crop_ax.imshow(self.t_h_crop_array[:,:,self.i],cmap='gray')
        self.t_h_crop_ax.set_aspect(0.25)
        self.create_t_h_crop_image(self.t_h_crop_array)
        self.r_coords_x_vals = []
        self.r_coords_y_vals = []
        self.t_h_select_point = False
        self.t_h_roll = True


    def t_h_lines(self):
        x0,y0 = self.r_coords_x_vals[0], self.r_coords_y_vals[0]
        x1,y1 = self.r_coords_x_vals[1], self.r_coords_y_vals[1]
        x2 = int((abs(x1-x0)/2)+x0)
        y2 = int(370)
        m1 = (y2-y0)/(x2-x0)
        m2 = (y2-y1)/(x2-x1)
        b1 = y0 - (m1*x0)
        b2 = y1 - (m2*x1)
        x_base_0 = int((-b1)/m1)
        x_base_1 = int((-b2)/m2)
        y_base_0 = 0
        y_base_1 = 0
        self.x_axis1 = np.linspace(x_base_0,x2,abs(x2-x_base_0))
        self.x_axis2 = np.linspace(x2,x_base_1,abs(x2-x_base_1))
        p0 = [x2,y2]
        p1 = [x_base_0,0]
        p2 = [x_base_1,0]
        x_vals1 = [x_base_0,x2]
        x_vals2 = [x2,x_base_1]
        y_vals1 = [0,y2]
        y_vals2 = [y2,0]
        coeff1 = np.polyfit(x_vals1,y_vals1,1)
        coeff2 = np.polyfit(x_vals2,y_vals2,1)
        poly1 = np.poly1d(coeff1)
        poly2 = np.poly1d(coeff2)
        self.y_axis1 = poly1(self.x_axis1)
        self.y_axis2 = poly2(self.x_axis2)
        self.t_h_fig.show()
        v_x0 = int(self.r_coords_x_vals[0])
        v_x1 = int(self.r_coords_x_vals[1])
        v_y0 = int(0)
        v_y1 = int(316)
        self.lines = self.t_h_crop_ax.plot(v_x0,v_y0, 'r-')
        self.lines.append(self.t_h_crop_ax.plot(v_x1,v_y1,'r-'))
        #self.lines = self.t_h_crop_ax.plot(self.x_axis1,self.y_axis1)
        #self.lines.append(self.t_h_crop_ax.plot(self.x_axis2,self.y_axis2))
        print('lines')
        print(x0,y0)
        print(x1,y1)
        print(x2,y2)
        print(x_base_0,x_base_1)
        self.t_h_crop_fig.canvas.draw()
        main.create_t_h_plot()

    def create_t_h_plot(self):
        self.t_h_plot_array = np.zeros((316,int(self.t_h_crop_array.shape[2])))
        x_vals1 = np.array(self.x_axis1)
        x_vals2 = np.flip(np.array(self.x_axis2))
        for i in range(self.t_h_plot_array.shape[1]):
            for j in range(316):
                #index = main.find_closest(self.y_axis1,j)
                #x0 = int(x_vals1[index])
                #x1 = int(x_vals2[index])
                #bkg1 = self.t_h_crop_array[j,0:x0,i]
                #bkg2 = self.t_h_crop_array[j,x1:,i]
                #bkg1 = bkg1.flatten()
                #bkg2 = bkg2.flatten()
                #bkg = np.concatenate((bkg1,bkg2))
                #bkg = np.average(bkg)
                #x0 = int(self.r_coords_x_vals[0])
                #x1 = int(self.r_coords_x_vals[1])
                #data = self.t_h_crop_array[j,x0:x1,i]
                x0,y0 = int(self.r_coords_x_vals[0]), int(self.r_coords_y_vals[0])
                x1,y1 = int(self.r_coords_x_vals[1]), int(self.r_coords_y_vals[1])
                data_slice = self.t_h_crop_array[j,x0:x1,i]
                bkg1 = self.t_h_crop_array[j,0:x0,i]
                bkg2 = self.t_h_crop_array[j,x1:,i]
                bkg1 = bkg1.flatten()
                bkg2 = bkg2.flatten()
                bkg = np.concatenate([bkg1,bkg2])
                bkg = np.average(bkg)
                data = data_slice - bkg
                data = np.average(data)
                std = np.std(data_slice)
                #data = data/std
                self.t_h_plot_array[j,i] = data
        #self.t_h_plot_array = np.diff(self.t_h_plot_array,axis=1)
        print(x_vals1)
        print(x_vals2)
        self.t_h_plot_fig.show()
        self.view_t_h_plot = self.t_h_plot_ax.imshow(self.t_h_plot_array, interpolation='None', cmap='magma',vmax=np.percentile(self.t_h_plot_array,90))
        self.t_h_plot_ax.set_aspect(0.15)


    def find_closest(self,array,value):
        array = np.asarray(array)
        index = (np.abs(array - value)).argmin()
        return index

    def on_press(self, event):
        self.coords_x0 = event.xdata
        self.coords_y0 = event.ydata
        self.r_coords_x_vals.append(self.coords_x0)
        self.r_coords_y_vals.append(self.coords_y0)
        self.coords_x_vals.append(self.coords_x0)
        self.coords_y_vals.append(self.coords_y0)
        if len(self.r_coords_x_vals) == 2:
            if self.t_h_select_point == True:
                print('crop')
                main.t_h_crop()
            elif self.t_h_roll == True:
                main.t_h_lines()
        else:
            pass

    def cmap_change(self):
        self.cmap_index = self.cmapSelect.currentIndex()
        self.current_cmap = self.cmaps[(self.cmap_index)]
        main.create_image(self.cube_display)

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
        self.cube_display = self.interval(self.cube_display)
        main.create_image(self.cube_display)
        print(self.interval)
        print(self.stretch_val)

# Contextual slider behavior, for stretch functions
    def setSliderValue(self):
        self.a = self.varSlider.value()
        self.rmmpl()
        self.create_image(self.cube_display)
        self.addmpl(self.fig1)

    def grid_lines(self, state):
        if state == Qt.Checked:
            self.axf1.grid(b=True)
        else:
            self.axf1.grid(b=False)
        main.create_image(self.cube_display)

    def t_h_div(self, cube):
        self.i = 0
        n = self.cube_display.shape[2]
        x = self.cube_display.shape[0]
        y = self.cube_display.shape[1]
        dataz = np.zeros((x,y,n))
        window = 5
        for i in range(n):
            if i < window:
                cube_1 = cube[:,:,0:(int(i+window))]
                window_means = np.mean(cube_1,axis=2)
                dataz[:,:,i] = window_means
            elif (n - i) < window:
                window_means = np.mean(cube[:,:,int(i-window):int(n)],axis=2)
                dataz[:,:,i] = window_means
            else:
                window_means = np.mean(cube[:,:,int(i-window):int(i+window)],axis=2)
                dataz[:,:,i] = window_means
        dataz = dataz + 0.03
        dataz = self.cube_display/dataz
        return dataz

    def cube_div(self, state):
        self.cube_base2 = self.cube_display
        if state == Qt.Checked:
            self.i = 0
            n = self.cube_display.shape[2]
            x = self.cube_display.shape[0]
            y = self.cube_display.shape[1]
            dataz = np.zeros((x,y,n))
            max_brightness = np.max(self.cube_display,axis=2)
            self.cube_display = self.cube_display / max_brightness
        else:
            self.cube_display = self.cube_base2
        main.create_image(self.cube_display)


    def exp_norm(self, state):
        if state == Qt.Checked:
            self.cube_expnorm = self.cube_display
            self.i = 0
            for i in range(self.n):
                data = fits.open(self.file_list[i], ignore_missing_simple=True)
                cmd_exp = data[1].header['EXPTIME']
                self.cube_display[:,:,i] = self.cube_expnorm[:,:,i] / cmd_exp
            main.create_image(self.cube_display)
        else: 
            self.cube_display = self.cube_base
            main.create_image(self.cube_display)


    def run_diff(self, state):
        if state == Qt.Checked:
            self.cube_nodiff = self.cube_display
            self.i = 0
            self.cube_display = np.diff(self.cube_display, axis=2)
            self.n = self.cube_display.shape[2]
            main.create_image(self.cube_display)
        else:
            self.cube_display = self.cube_nodiff
            self.n = self.cube_display.shape[2]
            main.create_image(self.cube_display)

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

    def set_ng_cube_size_x(self):
        self.ng_cube_size_x = int(self.ng_cube_size_x_txt.text())
        self.ng_cube_size_x_txt.clearFocus()
    
    def set_ng_cube_size_y(self):
        self.ng_cube_size_y = int(self.ng_cube_size_y_txt.text())
        self.ng_cube_size_x_txt.clearFocus()

    def set_ng_cube_size_t(self):
        self.ng_cube_size_t = int(self.ng_cube_size_t_txt.text())
        self.ng_cube_size_t_txt.clearFocus()

    def set_ng_cube_div(self):
        self.ng_cube_div = int(self.ng_cube_div_txt.text())
        self.ng_cube_div_txt.clearFocus()

    #def noisegate(self):
    #    self.backup_cube = self.cube_display
    #    self.cube_display = ng.noise_gate_batch(self.cube_display, cubesize=(self.ng_cube_size_t, self.ng_cube_size_y, self.ng_cube_size_x), cubediv = self.ng_cube_div)
    #    main.create_image(self.cube_display)


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
        elif self.cube_norm_on == True:
            image = self.cube_norm[:,:,i]
        elif self.cube_filter_on == True:
            image = self.cube_base[:,:,i]
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
        elif self.cube_norm_on == True:
            image_0 = self.cube_norm[:,:,i]
            a_0 = np.min(self.cube_norm)
            a_1 = np.max(self.cube_norm)
        elif self.cube_filter_on == True:
            image_0 = self.cube_base[:,:,i]
            a_0 = np.min(self.cube_base)
            a_1 = np.max(self.cube_base)
        mean_local_norm = np.mean(image_set,axis=2)
        gamma_transform = ((image_0 - a_0) / (a_1 - a_0))**(1.0 / self.msgn_gamma)
        msgn_image = (self.msgn_h * gamma_transform) + ((1 - self.msgn_h) * self.msgn_g * mean_local_norm)
        self.cube_filter[:,:,i] = msgn_image
        #self.cube_display = self.cube_filter
        #main.create_image(self.cube_display)

    def msgn_all_images(self):
        self.cube_filter = np.zeros((self.cube_x,self.cube_y,self.n))
        for i in tqdm.tqdm(range(self.n)):
            main.msgn_image_set(i)
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
        
    def sunkit_mgn(self):
        self.cube_filter = np.zeros((self.cube_x,self.cube_y,self.n))
        self.cube_filter = enhance.mgn(self.cube_base, sigma=[5,10,20,40])
        self.cube_display = self.cube_filter
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
        n = self.cube_base.shape[2]
        x = self.cube_base.shape[0]
        y = self.cube_base.shape[1]
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
            self.cube_display = self.stretch_val(self.cube_display)
            #self.norm = ImageNormalize(interval=self.interval, stretch=self.stretch_val, clip=True)
        elif text == 'Sqrt Stretch':
            self.cube_display = np.sqrt(self.cube_display)
        elif text == 'No Stretch':
            self.norm = None
            self.cube_display = self.cube_base
        else:
            self.var_max = 10
            self.var_int = 1
            self.cube_display = self.stretch_val(self.cube_base, clip=False)
            #self.norm = ImageNormalize(interval=self.interval, stretch=self.stretch_val, clip=True)
        main.create_image(self.cube_display)

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
            print('coordinates added')
        elif self.set_zoom_coords == True:
            self.zoom_coords.append([round(self.ix),round(self.iy)])
            if len(self.zoom_coords) == 2:
                print('Zoom')
                main.zoom()
        pix_value = self.cube_display[round(self.iy),round(self.ix),self.i]
        self.lcdVal.display(pix_value)
        self.lcdX.display(int(self.ix))
        self.lcdY.display(int(self.iy))

# Creates a 3-d ndarray from list of files and creates a
# list of fits headers
    def make_cube(self, files):
        n = len(files)
        image0 = fits.getdata(files[0])
        x = image0.shape[0]
        y = image0.shape[1]
        dataz = np.zeros((x,y,n))
        for i in range(n):
            data = fits.getdata(files[i], ignore_missing_simple=True)
            data[np.isnan(data)] = 0
            data[np.isinf(data)] = 0
            data[data < 0] = 0
            dataz[:,:,i] = data
            fits_data = fits.open(files[i], igmore_missing_simple=True)
            date_obs = fits_data[1].header['DATE-OBS']
            self.dates.append(date_obs)
        return dataz

    def zoom(self):
        self.backup_cube = self.cube_display
        self.hdu_backup = self.hdu
        self.wcs_backup = self.wcs
        x1,x2 = self.zoom_coords[0][0], self.zoom_coords[1][0]
        y1,y2 = self.zoom_coords[0][1], self.zoom_coords[1][1]
        self.cube_display = self.cube_display[y2:y1,x1:x2,:]
        print(self.cube_display.shape)
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
        print('Undo Zoom')
        #self.hdu = self.hdu_backup
        #self.wcs = self.wcs_backup
        #self.axf1 = self.canvas.figure.subplots(ncols=1,nrows=1, subplot_kw={'projection': self.wcs})
        main.create_image(self.cube_display)

    def zoom_state(self, checked):
        if checked:
            self.set_zoom_coords = True
            print('Zoom checked')
        else:
            self.set_zoom_coords = False
            print('Zoom unchecked')
        

    def import_session(self):
        file_name = QFileDialog.getOpenFileName(self,'Open Session File')
        session_file = h5py.File(file_name[0],'r')
        mosaic_cube_exists = False
        if 'cube_mosaic' in session_file:
            self.mosaic_cube = session_file['cube_mosaic'][:]
            mosaic_cube_exists = True
        else:
            pass
        self.cube_base = session_file['cube_base'][:]
        if len(self.file_list) == 0:
            if os.path.exists('file_list.txt') == True:
                file_list_temp = open('file_list.txt').readlines()
                for i in file_list_temp:
                    self.file_list.append(i.strip())
            else:
                pass
        else:
            pass
        if mosaic_cube_exists == True:
            if len(self.mosaic_final_list_1) == 0:
                if os.path.exists('matched_files2.txt') == True:
                    mosaic_final_list_1_temp = open('matched_files1.txt').readlines()
                    for i in mosaic_final_list_1_temp:
                        self.mosaic_final_list_1.append(i.strip())
                else:
                    pass
            else:
                pass
            if len(self.mosaic_final_list_2) == 0:
                if os.path.exists('matched_files2.txt') == True:
                    mosaic_final_list_2_temp = open('matched_files2.txt').readlines()
                    for i in mosaic_final_list_2_temp:
                        self.mosaic_final_list_2.append(i.strip())
                else:
                    pass
            else:
                pass
            if len(self.mosaic_files_list_1) == 0:
                if os.path.exists('mosaic_files1.txt') == True:
                    mosaic_files_list_1_temp = open('mosaic_files1.txt').readlines()
                    for i in mosaic_files_list_1_temp:
                        self.mosaic_files_list_1.append(i.strip())
                else:
                    pass
            else:
                pass
            if len(self.mosaic_files_list_2) == 0:
                if os.path.exists('mosaic_files2.txt') == True:
                    mosaic_files_list_2_temp = open('mosaic_files2.txt').readlines()
                    for i in mosaic_files_list_2_temp:
                        self.mosaic_files_list_2.append(i.strip())
                else:
                    pass
            else:
                pass
        else:
            pass
        if mosaic_cube_exists == True:
            self.hdu = fits.open(self.mosaic_files_list_1[0])[0]
            n = len(self.mosaic_files_list_2)
        else:
            self.hdu = fits.open(self.file_list[0])[0]
            n = len(self.file_list)
        self.wcs = WCS(self.hdu.header)
        self.axf1 = self.canvas.figure.subplots(ncols=1,nrows=1, subplot_kw={'projection': self.wcs})
        if mosaic_cube_exists == True:    
            for i in range(n):
                fits_data = fits.open(self.mosaic_files_list_2[i])
                date_obs = fits_data[0].header['DATE-OBS']
                self.dates.append(date_obs)
                self.cube_display = self.mosaic_cube
        else:
            for i in range(n):
                fits_data = fits.open(self.file_list[i])
                date_obs = fits_data[1].header['DATE-OBS']
                self.dates.append(date_obs)
                self.cube_display = session_file['cube_display'][:]
        self.image = self.cube_display[:,:,0]
        self.view_image = self.axf1.imshow(self.image)
        self.n = self.cube_display.shape[2]
        main.create_image(self.cube_display)

    def save_session(self):
        cwd = os.getcwd()
        file_path = QFileDialog.getSaveFileName(self,"Save Session","","*.hdf5")
        h5_file = h5py.File(file_path[0],'w')
        h5_file.create_dataset('cube_display',data = self.cube_display)
        h5_file.create_dataset('cube_base',data = self.cube_base)
        if self.mosaic_cube != None:
            h5_file.create_dataset('cube_mosaic',data = self.mosaic_cube)
        else:
            pass
        h5_file.create_dataset('file_list',data = self.file_list)
        h5_file.close()
        filelist_file = open('file_list.txt','w')
        for element in self.file_list:
            filelist_file.write(element + '\n')
        filelist_file.close()
        matched_files1 = open('matched_files1.txt','w')
        for i in self.mosaic_final_list_1:
            matched_files1.write(i + '\n')
        matched_files1.close()
        matched_files2 = open('matched_files2.txt','w')
        for j in self.mosaic_final_list_2:
            matched_files2.write(j + '\n')
        matched_files2.close()
        mosaic_files1 = open('mosaic_files1.txt','w')
        for a in self.mosaic_files_list_1:
            mosaic_files1.write(a + '\n')
        mosaic_files1.close()
        mosaic_files2 = open('mosaic_files2.txt','w')
        for b in self.mosaic_files_list_2:
            mosaic_files2.write(b + '\n')
        mosaic_files2.close()
        

    def save_files(self):
        temp_name = self.file_list[0]
        path,file_temp = os.path.split(temp_name)
        today = datetime.now().strftime('%Y%m%d%H%M')
        os.mkdir(path + '/' + today)
        new_path = path + '/' + today
        print(str(new_path))
        for f in range(self.n):
            current_name = self.file_list[f]
            new_name = Path(current_name).stem
            new_filepath = new_path + '/' + new_name + '_new.fits'
            open_file = fits.open(current_name)
            data = self.cube_display[:,:,f]
            open_file[0].data = data
            open_file.writeto(new_filepath, overwrite=True)

    def save_movie(self):
        cwd = os.getcwd()
        self.images_folder = os.path.join(cwd,'movie_temp')
        if os.path.isdir(self.images_folder) == True:
            shutil.rmtree(self.images_folder)
        os.makedirs(self.images_folder)
        for i in range(self.n):
            date = self.dates[i]
            self.view_image.set_norm(self.norm)
            self.image = np.flip(self.cube_display[:,:,i],axis=0)
            imageMin = np.amin(self.image)
            imageMax = np.amax(self.image)
            self.view_image.set_data(self.image)
            self.axf1.set_title(date)
            self.fig1.canvas.draw_idle()
            self.lcdMin.display(imageMin)
            self.lcdMax.display(imageMax)
            image_name = 'image'+str(i)+'.png'
            file_name = os.path.join(self.images_folder,image_name)
            plt.imsave(file_name, self.image, cmap = 'goes-rsuvi131')
            #self.fig1.canvas.figure.savefig(file_name)
        print(self.images_folder)
        file_path = QFileDialog.getSaveFileName(self,"Save Video","","Video (*.mp4)")
        video_name = file_path[0].rsplit('/',1)[1]
        image_files = [f for f in os.listdir(self.images_folder) if os.path.isfile(os.path.join(self.images_folder,f))]
        images = []
        for i in image_files:
            image_file = self.images_folder + '/' + i
            images.append(image_file)
        images = natsorted(images)
        print(images)
        frame = cv2.imread(images[0])
        height,width,layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video=cv2.VideoWriter(video_name, fourcc, 5, (width,height))
        for j in images:
            img = cv2.imread(j)
            video.write(img)
        cv2.destroyAllWindows()
        video.release()
        print('Saved '+str(video_name))

        #shutil.rmtree(images_folder)

    def pull_files(index1,index2):
        d = self.images_folder
        myfiles = []
        files = [f for f in listdir(d) if isfile(join(d, f))]
        for i in files:
            if index1.lower() in i.lower() and index2.lower() in i.lower():
                myfiles.append(i)
        myfiles = natsorted(myfiles)
        return myfiles


# Creates individual image to be displayed and generates
# accompanying date from header
    def create_image(self,cube):
        date = self.dates[self.i]
        self.view_image.set_norm(self.norm)
        self.image = cube[:,:,self.i]
        #imageMin = int(np.amin(self.image))
        #imageMax = int(np.amax(self.image))
        self.view_image.set_data(self.image)
        self.view_image.set_cmap(self.current_cmap)
        self.axf1.set_title(date)
        self.fig1.canvas.draw_idle()
        #self.lcdMin.display(imageMin)
        #self.lcdMax.display(imageMax)
    
    def create_t_h_image(self,cube):
        print(self.t_h_i)
        date = self.dates[self.t_h_i]
        self.view_t_h_image.set_norm(self.norm)
        self.image = cube[:,:,self.t_h_i]
        self.view_t_h_image.set_data(self.image)
        self.view_t_h_image.set_cmap(self.current_cmap)
        self.t_h_ax.set_title(date)
        self.t_h_fig.canvas.draw()

    def create_t_h_crop_image(self,cube):
        print(self.t_h_crop_i)
        date = self.dates[self.t_h_crop_i]
        self.view_t_h_crop.set_norm(self.norm)
        self.image = cube[:,:,self.t_h_crop_i]
        self.view_t_h_crop.set_data(self.image)
        self.t_h_crop_ax.set_title(date)
        self.t_h_crop_fig.canvas.draw()

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
        self.file_list = sorted(files)
        self.hdu = fits.open(self.file_list[0])[1]
        self.wcs = WCS(self.hdu.header)
        self.axf1 = self.canvas.figure.subplots(ncols=1,nrows=1, subplot_kw={'projection': self.wcs})
        self.cube_base = main.make_cube(self.file_list)
        #self.cube_base = sunpy.map.Map(self.file_list,sequence=True)
        #self.cube_base = mc_coalign(self.cube_base)
        #self.cube_base = self.cube_base.as_array()
        self.cube_display = self.cube_base
        self.cube_base2 = self.cube_base
        self.cube_x = self.cube_base.shape[0]
        self.cube_y = self.cube_base.shape[1]
        self.cube_n = self.cube_base.shape[2]
#        self.cube = self.cube[725:825,950:1280,:]
        self.n = self.cube_base.shape[2]
        self.i = 0
        self.cube_base_on = True
        self.image = self.cube_display[:,:,0]
        self.view_image = self.axf1.imshow(self.image)
        self.fig1.canvas.draw_idle()

# Initializes this application when python file is run
if __name__ == '__main__':

    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

