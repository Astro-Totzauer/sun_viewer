import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import time,glob
import os
import sys, random
import time
from pytest import skip
import fitscal as fc
import keyboard
from os import listdir,walk
import sunkit_image.enhance as enhance
from os.path import isfile,join
import matplotlib
import savitzkygolay
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure 
import matplotlib.pyplot as plt
import scipy as sp1
import scipy.ndimage
from scipy import signal
from astropy.visualization import ContrastBiasStretch,ManualInterval,LinearStretch,MinMaxInterval,ImageNormalize, SqrtStretch,LogStretch,PowerDistStretch,PowerStretch,SinhStretch,SquaredStretch,AsinhStretch,PercentileInterval
from scipy.ndimage.interpolation import shift
from natsort import natsorted
import shutil
import sunpy.map
import sunpy.io.fits as sunfits
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from astropy.coordinates import SkyCoord
from astropy import units as u
import sunpy.image.coalignment as sunpy_coalign
import savitzkygolay
import sunpy.visualization.animator as sunpy_anomator
import h5py
#import noisegate.tools as ng


def listfiles(index):
    d = os.getcwd()+'/'
    myfiles = []
    files = [f for f in listdir(d) if isfile(join(d, f))]
    for i in files:
        if index in i:
            myfiles.append(d+i)
    return myfiles

def getimage(x):
    d = os.getcwd()+'/'
    file = files(x,'.fits')[0] 
    image = fits.getdata(file)
    return image

def image_fits(x):
    d = os.getcwd()+'/'
    file = d+x
    image = fits.getdata(file)
    return image

def pulltimes(files):
    times = []
    n = len(files)
    sortedfiles = natsorted(files)
    data0 = fits.open(sortedfiles[0])
    data0time = data0[0].header['DATE-OBS']
    time0 = Time(data0time,format='fits')
    jdtime0 = time0.jd
    for i in range(n):
        data = fits.open(sortedfiles[i])
        datetime = data[0].header['DATE-OBS']
        time = Time(datetime,format='fits')
        jdtime = (time.jd - jdtime0)*86400
        times.append(jdtime)
    return times

def obsdate(index):
    d = os.getcwd()+'/'
    file = files(d,index,'.fits')[0]
    data = fits.open(file)
    dateobs = data[0].header['DATE-OBS']
    return dateobs

def imstretch_suvi(image,cmap_input):
    image = np.flip(image,axis=0)
    med = np.median(image)
    std = np.std(image)
    vminimum = med-std
    vmaximum = med+std
    plt.imshow(image,vmin=vminimum,vmax=vmaximum, cmap = str(cmap_input))
    plt.show()

def exptime(x):
    d = os.getcwd() + '/'
    file = files(d,x,'.fits')[0]
    data = fits.open(file)
    time = data[0].header['exptime']
    return time

def header(file):
    hdu1 = fits.open(file)
    hdr = hdu1[0].header
    print(hdr)

def uniquetimes(x):
    d = os.getcwd() + '/'
    filelist = files(x,'.fits')
    times = []
    for i in filelist:
        print(i)
        data = fits.open(i, ignore_missing_simple=True)
        cmd_exp = data[1].header['EXPTIME']
        times.append(cmd_exp)
    return np.unique(times)   

def separate_times(files, x):
    for i in files:
        data = fits.open(i, ignore_missing_simple=True)
        cmd_exp = data[1].header['EXPTIME']
        if cmd_exp < x == True:
            new_name = 'short_'+str(i)
            os.rename(i, new_name)
        else:
            new_name = 'long_'+str(i)  
            os.rename(i, new_name)
        print('Files renamed')

def suvitimes(x):
    d = os.getcwd() + '/'
    filelist = files('SUVI','fits')
    files1 = []
    for i in filelist:
        data = fits.open(i)
        cmd_exp = data[0].header['CMD_EXP']
        if cmd_exp == x:
            files1.append(i)
    return files1

def suvitimes_files(x,y,t):
    file_list = files(x,y)
    file_list2 = []
    for i in file_list:
        data = fits.open(i)
        t_0 = data[0].header['CMD_EXP']
        if t_0 == t:
            file_list2.append(i)
    return file_list2

def get_suvi_date(file):
    d = os.getcwd() + '/'
    data = fits.open(file)
    date_obs = data[0].header['DATE-OBS']
    return date_obs

def wavelength(x):
    d = os.getcwd() + '/'
    file = files(d,x,'.fits')[0]
    data = fits.open(file)
    wavelength = data[0].header['WAVELNTH']
    return wavelength

# Pulls the filter from a file's fits header

def filt(x):
    d = os.getcwd()+'/'
    file = files(d,x,'.fits')[0]
    data = fits.open(file)
    filtinfo = data[0].header['filter']
    return filtinfo

def files(x,y):
    d = os.getcwd()+'/'
    myfiles = []
    files = [f for f in listdir(d) if isfile(join(d, f))]
    for i in files:
        if x.lower() in i.lower() and y.lower() in i.lower():
            myfiles.append(i)
    return myfiles

def rename(x,y):
    d = os.getcwd()+'/'
    for i in glob.glob(d+'*.*'):
        new_filename = i.replace(x,y)
        os.rename(i,new_filename) 

def rename2(d,x,y):
    for i in glob.glob(d+'*.*'):
        new_filename = i.replace(x,y)
        os.rename(i,new_filename) 

def deletefiles(x):
    d = os.getcwd()+'/'
    for file in glob.glob(d+'*.*'):
        if x in file:
            os.remove(file)

# Moves files with keywords x and y from location d1 to location d2

def move(d1,d2,x,y):
    filelist = files(x,y)
    for i in filelist:
        shutil.move(i,d2+os.path.basename(i))
    return 'Files Moved'

def imstretch_suvi(image,cmap_input):
    image = np.flip(image,axis=0)
    med = np.median(image)
    std = np.std(image)
    vminimum = med-std
    vmaximum = med+std
    plt.imshow(image,vmin=vminimum,vmax=vmaximum, cmap = str(cmap_input))
    plt.show()

def imstretch_suvi_2(file,cmap_input):
    data = fits.getdata(file)
    image = np.flip(data,axis=0)
    med = np.median(image)
    std = np.std(image)
    vminimum = med-std
    vmaximum = med+std
    plt.imshow(image,vmin=vminimum,vmax=vmaximum, cmap = str(cmap_input))

def move_files(d,file_list):
    for i in file_list:
        shutil.move(i,d)
    return 'Files Moved'

def mcslice(files):
    n = len(files)
    for i in range(n):
        image,hdr = fits.getdata(files[i],header=True)
        data = image.data
        date_raw = str(hdr['DATE-OBS'])
        date_index = date_raw.index('T')
        date_simp = date_raw[:date_index]
        date_simp = date_simp.replace('-','')
        time = date_raw[date_index+1:]
        time = time.replace(':','')
        time = time.replace('.','')
        time = time[:-3]
        wavelength = str(hdr['WAVELNTH'])
        wavelength_index = wavelength.index('.')
        wavelength = wavelength[:wavelength_index]
        dslice = image[650:937,985:1272]
        filename = 'SUVI_'+wavelength+'_'+date_simp+'_'+time+'_'+'Cropped_Gated.fits'
        fits.writeto(filename,dslice,header=hdr)


def mcslice2(mapsequence):
    n = len(mapsequence)
    for i in range(n):
        data = mapsequence[i]
        top_right = SkyCoord(1490.0*u.arcsec, -190.0*u.arcsec, frame = data.coordinate_frame)
        bottom_left = SkyCoord(875.0*u.arcsec, -210.0*u.arcsec, frame = data.coordinate_frame)
        data_submap = data.submap(bottom_left, top_right=top_right)
        date_raw = str(data.date)
        date_index = date_raw.index('T')
        date_simp = date_raw[:date_index]
        date_simp = date_simp.replace('-','')
        time = date_raw[date_index+1:]
        time = time.replace(':','')
        time = time.replace('.','')
        time = time[:-3]
        wavelength = str(data.wavelength)
        wavelength_index = wavelength.index('.')
        wavelength = wavelength[:wavelength_index]
        data_submap.save('SUVI_'+wavelength+'_'+date_simp+'_'+time+'_'+'Cropped.fits')


def imcube(images):
    n = len(images)
    image0 = image_fits(images[0])
    x = image0.shape[0]
    y = image0.shape[1]
    dataz = np.zeros((x,y,n))
    for i in range(n):
        dataz[:,:,i] = image_fits(images[i])
    return dataz

def image_play(data):
    index = 0
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(bottom=0.15)
    im_h = imstretch_suvi_2(data[:,:,index],'gray')
    ax_depth = plt.axes([0.23,0.02,0.56,0.04])
    slider_depth = Slider(ax_depth, 'depth', 0, data.shape[2]-1, valinit=index)

    def update_depth(val):
        index = int(round(slider_depth.val))
        im_h.set_data(imstretch_suvi_2(data[:,:,index],'gray'))
    slider_depth.on_changed(update_depth)
    plt.show()

def change_obsgeo(files):
    for i in files:
        fits.setval(i, 'OBSGEO-X', value=0)
        fits.setval(i, 'OBSGEO-Y', value=0)
        fits.setval(i, 'OBSGEO-Z', value=0)
    
def sunpy_align_files(files):
    map_sequence = sunpy.map.Map(files, sequence=True)
    shifts = sunpy_coalign.calculate_match_template_shift(map_sequence, layer_index=0,)
    x_shifts, y_shifts = shifts['x'], shifts['y']
    x_shifts = (np.array(x_shifts/2.5))*u.pix
    y_shifts = (np.array(y_shifts/2.5))*u.pix
    map_sequence_aligned = sunpy_coalign.apply_shifts(map_sequence, y_shifts, x_shifts)
    n = len(map_sequence_aligned)
    for i in range(n):
        data = map_sequence_aligned[i]
        date_raw = str(data.date)
        date_index = date_raw.index('T')
        date_simp = date_raw[:date_index]
        date_simp = date_simp.replace('-','')
        time = date_raw[date_index+1:]
        time = time.replace(':','')
        time = time.replace('.','')
        time = time[:-3]
        wavelength = str(data.wavelength)
        wavelength_index = wavelength.index('.')
        wavelength = wavelength[:wavelength_index]
        data.save('SUVI_'+wavelength+'_'+date_simp+'_'+time+'Aligned.fits')

def savitzkygolay_smooth(files,window,poly):
    files = sorted(files)
    data = []
    for fn in files:
        with fits.open(fn) as f:
            data.append(f[0].data.copy())
    data = np.array(data)
    smoothed = savitzkygolay.filter3D(data, window, poly)
    #for i, fn in enumerate(files):
        #data2,hdr = fits.getdata(fn,header=True)
        #new_fn = os.path.basename(fn).replace('.fits','_filtered.fits')
        #ifits.writeto(new_fn, smoothed[i,:,:],header=hdr)
    return data, smoothed


def sg_display(images1,images2,n,title1,title2):
    image1 = np.flip(images1[n],axis=0)
    image2 = np.flip(images2[n],axis=0)
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    norm1 = ImageNormalize(image1, interval=ManualInterval(vmin=0.25,vmax=1),stretch=ContrastBiasStretch(0.3,0.5))
    norm2 = ImageNormalize(image2, interval=ManualInterval(vmin=0.25,vmax=1),stretch=ContrastBiasStretch(0.3,0.5))
    axes[0].imshow(image1,norm=norm1,cmap='gray'),axes[0].set_title(title1)
    axes[1].imshow(image2,norm=norm2,cmap='gray'),axes[1].set_title(title2)
    fig.show()

def s_g_reduction(data,smoothed):
    data_means = (fc.cube_mean(data,5))
    data_div = data / data_means
    smoothed_means = (fc.cube_mean(smoothed,5))
    smoothed_div = smoothed / smoothed_means
    return data_div, smoothed_div

def mapsequence_norm_edit(mapsequence):
    n = len(mapsequence)
    for i in range(n):
        mapsequence[i].plot_settings['norm'] = ImageNormalize(stretch=ContrastBiasStretch(0.35,0.7),vmin=0.15,vmax=0.85)

def adjust_image(cube):
    n = cube.shape[2]
    i = 0
    fig = plt.figure()
    image = np.flip(cube[:,:,i],axis=0)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)
    im = ax.imshow(image,cmap='gray')
    axcolor = 'lightgoldenrodyellow'
    axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_im = fig.add_axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_sqrt = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    b_sqrt = Button(ax_sqrt, 'SqrtStretch')
    smin = Slider(axmin, 'V_min', 0, 1, valinit=0,valstep=0.1)
    smax = Slider(axmax, 'V_max', 0, 1, valinit=1,valstep=0.1)
    s_im = Slider(ax_im, 'Images', 0, n, valinit=0, valstep=1)
    def sqrt_stretch(val):
        stretch_val = SqrtStretch(v_min,v_max)
        norm = ImageNormalize(image,stretch=stretch_val)
    def update(val):
        image = np.flip(cube[:,:,i],axis=0)
        norm = ImageNormalize(image,stretch=ContrastBiasStretch(smin.val,smax.val))
        im = ax.imshow(image,cmap='gray')
        im.set_norm(norm)
        fig.canvas.draw()
    def update_image(val):
        i = int(s_im.val)
        image = np.flip(cube[:,:,i],axis=0)
        im = ax.imshow(image,cmap='gray')
        fig.canvas.draw()
    def key_event(event):
        if event.key == 'left':
            s_im.set_value(s_im.value - 1)
        elif event.key == 'right':
            s_im.set_value(s_im.value + 1)
        else:
            pass
    smin.on_changed(update)
    smax.on_changed(update)
    s_im.on_changed(update_image)
    b_sqrt.on_clicked(sqrt_stretch)
    plt.show()

def show_image(cube):
    n = cube.shape[2]
    image = np.flip(cube[:,:,0],axis=0)
    norm = ImageNormalize(image, stretch=ContrastBiasStretch(0,1))
    fig, ax = plt.subplots()
    axcolor = 'lightgoldenrodyellow'
    axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_im = fig.add_axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_sqrt = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    b_sqrt = Button(ax_sqrt, 'SqrtStretch')
    smin = Slider(axmin, 'V_min', 0, 1, valinit=0,valstep=0.1)
    smax = Slider(axmax, 'V_max', 0, 1, valinit=1,valstep=0.1)
    s_im = Slider(ax_im, 'Images', 0, n, valinit=0, valstep=1)
    ax.imshow(image, norm=norm, cmap='gray')

    class controls(object):
        i = int(0)
        v_min = 0
        v_max = 1

        def display(self):
            image = np.flip(cube[:,:,self.i],axis=0)
            norm = ImageNormalize(image, stretch = ContrastBiasStretch(self.v_min,self.v_max))
            im = ax.imshow(image,cmap='gray')
            im.set_norm(norm)
            fig.canvas.draw()

   
        #def sqrt_stretch(self,val):
         #   self.stretch_val = SqrtStretch(self.v_min, self.v_max)
          #  norm = ImageNormalize(image,stretch=self.stretch_val)
           # display()

        def update(self,val):
            self.v_min = smin.val
            self.v_max = smax.val
            display()

        def update_image(self,val):
            self.i = int(s_im.val)
            display()

    smin.on_changed(controls.update)
    smax.on_changed(controls.update)
    s_im.on_changed(controls.update_image)
    #b_sqrt.on_clicked(controls.sqrt_stretch)
    plt.show()

def test_image(cube):

    def key(event):
        if event.key == 'left':
            im_prev(event)
        elif event.key == 'right':
            im_next(event)

    n = cube.shape[2]
    data_means = (fc.cube_mean(cube,5))
    data_div = cube / data_means
    i = 0
    v_min = 0
    v_max = 1
    norm = ImageNormalize(stretch=ContrastBiasStretch(v_min,v_max))
    image = np.flip(cube[:,:,0],axis=0)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.25, bottom=0.25)
    fig.canvas.mpl_connect('key_press_event', key)
    axcolor = 'lightgoldenrodyellow'
    axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_im = fig.add_axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_stretch = fig.add_axes([0.05,0.35,0.2, 0.2])
    ax_prev = fig.add_axes([0.25,0.05, 0.08, 0.03])
    ax_next = fig.add_axes([0.82,0.05, 0.08, 0.03])
    ax_div = fig.add_axes([0.05, 0.60,0.2,0.2])
    b_stretch = RadioButtons(ax_stretch, ('ContrastBias','Sqrt','Linear', 'Log', 'Power', 'PowerDist'))
    b_div = RadioButtons(ax_div, ('No Average', 'Average'))
    smin = Slider(axmin, 'V_min', 0, 1, valinit=0,valstep=0.1)
    smax = Slider(axmax, 'V_max', 0, 1, valinit=1,valstep=0.1)
    s_im = Slider(ax_im, 'Images', 0, n, valinit=0, valstep=1)
    b_prev = Button(ax_prev,'Prev')
    b_next = Button(ax_next,'Next')
    im = ax.imshow(image, cmap='gray')
    im.set_norm(norm)
    fig.canvas.draw()

    def im_prev(event):
        s_im.set_val(s_im.val -1)

    def im_next(event):
        s_im.set_val(s_im.val + 1)

    def div(label):
        global norm
        global image
        global im
        global i
        global data_div
        d0 = np.flip(cube[:,:,i],axis=0)
        d1 = np.flip(data_div[:,:,i],axis=0)
        div_dict = {'No Average': d0, 'Average':d1}
        image = div_dict[label]
        im = ax.imshow(image, cmap='gray')
        im.set_norm(norm)
        fig.canvas.draw()

    def stretch_function(label):
        global norm
        global image
        global im
        global i
        f0 = ContrastBiasStretch(v_min,v_max)
        f1 = SqrtStretch()
        f2 = LinearStretch()
        f3 = LogStretch()
        f4 = PowerStretch(2)
        f5 = PowerDistStretch()
        stretch_dict = {'ContrastBias': f0, 'Sqrt': f1, 'Linear': f2, 'Log': f3, 'Power': f4, 'PowerDist': f5}
        norm = ImageNormalize(stretch=stretch_dict[label])
        image = np.flip(cube[:,:,i],axis=0)
        im = ax.imshow(image, cmap='gray')
        im.set_norm(norm)
        fig.canvas.draw()

    def update_v(val):
        global v_min
        global v_max
        global norm
        global i
        v_min = smin.val
        v_max = smax.val
        norm = norm
        image = np.flip(cube[:,:,i],axis=0)
        im = ax.imshow(image,cmap='gray')
        im.set_norm(norm)
        fig.canvas.draw()

    def update_image(val):
        global i
        global norm
        i = int(s_im.val)
        image = np.flip(cube[:,:,i],axis=0)
        norm = norm
        im = ax.imshow(image,cmap='gray')
        im.set_norm(norm)
        fig.canvas.draw()

    b_prev.on_clicked(im_prev)
    b_next.on_clicked(im_next)
    s_im.on_changed(update_image)
    smin.on_changed(update_v)
    smax.on_changed(update_v)
    b_stretch.on_clicked(stretch_function)
    b_div.on_clicked(div)
    plt.show()


def test_images(cube1, cube2):

    def key(event):
        if event.key == 'left':
            im_prev(event)
        elif event.key == 'right':
            im_next(event)

    n = cube1.shape[0]
    a = 100
    i = 0
    v_min = 0
    v_max = 1
    stretch_val = SqrtStretch()
    norm = ImageNormalize(interval=MinMaxInterval(),stretch=stretch_val)
    image1 = np.flip(cube1[:,:,0],axis=0)
    image2 = np.flip(cube2[:,:,0],axis=0)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.set_title('Original')
    ax2.set_title('Normalized')
    fig.subplots_adjust(left=0.25, bottom=0.25)
    fig.canvas.mpl_connect('key_press_event', key)
    axcolor = 'lightgoldenrodyellow'
    axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_im = fig.add_axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_stretch = fig.add_axes([0.03,0.35,0.2, 0.2])
    ax_prev = fig.add_axes([0.25,0.05, 0.08, 0.03])
    ax_next = fig.add_axes([0.82,0.05, 0.08, 0.03])
    ax_div = fig.add_axes([0.03, 0.60,0.2,0.2])
    #ax_txt = fig.add_axes([0.03, 0.2,0.15,0.075])
    #text_box = TextBox(ax_txt, initial='100')
    b_stretch = RadioButtons(ax_stretch, ('ContrastBias','Sqrt','Linear', 'Log', 'Power', 'PowerDist'))
    b_div = RadioButtons(ax_div, ('No Average', 'Average'))
    smin = Slider(axmin, 'V_min', 0, 1, valinit=0,valstep=0.1)
    smax = Slider(axmax, 'V_max', 0, 1, valinit=1,valstep=0.1)
    s_im = Slider(ax_im, 'Images', 0, n, valinit=0, valstep=1)
    b_prev = Button(ax_prev,'Prev')
    b_next = Button(ax_next,'Next')
    im1 = ax1.imshow(image1, cmap='gray')
    im2 = ax2.imshow(image2, cmap='gray')
    im2.set_norm(norm)
    fig.canvas.draw()

    def im_prev(event):
        s_im.set_val(s_im.val -1)

    def im_next(event):
        s_im.set_val(s_im.val + 1)

    def div(label):
        global norm
        global image2
        global im2
        global i
        data_means = fc.cube_mean(cube2,5)
        data_div = cube2 / data_means
        d0 = np.flip(cube2[:,:,i],axis=0)
        d1 = np.flip(data_div[:,:,i],axis=0)
        div_dict = {'No Average': d0, 'Average':d1}
        image2 = div_dict[label]
        im2 = ax2.imshow(image2, cmap='gray')
        im2.set_norm(norm)
        fig.canvas.draw()


    def stretch_function(label):
        global stretch_val
        global image2
        global im2
        global i
        global v_min
        global v_max
        global norm
        global a
        f0 = ContrastBiasStretch(v_min,v_max)
        f1 = SqrtStretch()
        f2 = LinearStretch()
        f3 = LogStretch()
        f4 = PowerStretch()
        f5 = PowerDistStretch()
        stretch_dict = {'ContrastBias': f0, 'Sqrt': f1, 'Linear': f2, 'Log': f3, 'Power': f4, 'PowerDist': f5}
        stretch_val = stretch_dict[label]
        image2 = np.flip(cube2[:,:,i],axis=0)
        norm = ImageNormalize(interval=MinMaxInterval(),stretch=stretch_val)
        im2 = ax2.imshow(image2, cmap='gray')
        im2.set_norm(norm)
        fig.canvas.draw()

    def update_v(val):
        global v_min
        global v_max
        global norm
        global i
        global im2
        v_min = smin.val
        v_max = smax.val
        image2 = np.flip(cube2[:,:,i],axis=0)
        norm = ImageNormalize(interval=MinMaxInterval(),stretch=stretch_val)
        im2 = ax2.imshow(image2,cmap='gray')
        im2.set_norm(norm)
        fig.canvas.draw()

    def update_image(val):
        global i
        global norm
        i = int(s_im.val)
        image1 = np.flip(cube1[:,:,i],axis=0)
        image2 = np.flip(cube2[:,:,i],axis=0)
        norm = norm
        im1 = ax1.imshow(image1,cmap='gray')
        im2 = ax2.imshow(image2,cmap='gray')
        im2.set_norm(norm)
        fig.canvas.draw()

    b_prev.on_clicked(im_prev)
    b_next.on_clicked(im_next)
    s_im.on_changed(update_image)
    smin.on_changed(update_v)
    smax.on_changed(update_v)
    b_stretch.on_clicked(stretch_function)
    b_div.on_clicked(div)
    plt.show()

def cube_show(cube):
    n = cube.shape[2]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    

class image_display():
    def __init__(self, cube):
        self.cube = cube[2100:2600,0:500,:]
        self.n = self.cube.shape[2]
        self.n_2 = None
        self.i = 0
        self.rect = Rectangle((0,0), 1, 1,fill=None)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.max_ints = []
        self.points = None
        self.x_vals = []
        self.y_vals = []
        self.coords = []
        self.fig, self.ax = plt.subplots()
        self.image = np.flip(self.cube[:,:,0],axis=0)
        self.view_image = self.ax.imshow(self.image)
        self.fig.canvas.draw_idle()
        self.ax.set_title(self.i)
        self.ax.add_patch(self.rect)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid3 = self.fig.canvas.mpl_connect('button_release_event', self.on_release)


    def keyPressEvent(self, event):
        if event.key == 'left':
            self.prev()
        elif event.key == 'right':
            self.next()

    def prev(self):
        if self.i == 0:
            self.i = 0
        else:
            self.i = self.i - 1
        self.create_image()

    def next(self):
        if self.i == self.n - 1:
            self.i = self.n - 1
        else:
            self.i = self.i + 1
        self.create_image()

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        if self.points == None:
            print('1')
            pass
        else:
            print('2')
            for i in self.points:
                i.remove()
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.fig.canvas.draw()
        self.cube_new = self.cube[round(self.y0):round(self.y1),round(self.x0):round(self.x1),:]
        self.max_ints = np.argmax(self.cube_new[:,:,self.i], axis = 0)
        self.x_vals = np.arange(0,self.cube_new.shape[1]) + round(self.x0)
        self.y_vals = round(self.y1) - self.max_ints
        #self.points = np.dstack((self.x_coords,self.y_vals))[0]
        self.points = self.ax.plot(self.x_vals,self.y_vals, color='red')
        self.fig.canvas.draw()
        print(self.max_ints.shape)
        print(self.x_vals.shape)

            

    def create_image(self):
        self.ax.set_title(self.i)
        self.image = np.flip(self.cube[:,:,self.i],axis=0)
        self.view_image.set_data(self.image)
        self.fig.canvas.draw_idle()


class select_structure():
    def __init__(self, cube1):
        self.cube1 = cube1[550:675,0:500,145:270]
        #self.cube2 = cube2[700:850, 970:1280,:]
        self.n = self.cube1.shape[2]
        self.n_2 = None
        self.i = 0
        self.coords_file = h5py.File('t_h_plot_coords_20221107151547','r')
        self.coords_from_file = self.coords_file['x_coords_all']
        self.rect = Rectangle((0,0), 1, 1,fill=None)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.line1 = None
        self.line2 = None
        self.line_1 = None
        self.line_2 = None
        self.max_ints = []
        self.sheet_cube = None
        self.sheet2 = None
        self.n_y = None
        self.fit = None
        self.points = None
        self.pix_val = None
        self.y = 0
        self.z = None
        self.f = None
        self.x_vals = []
        self.y_vals = []
        self.coords = []
        self.fig, self.ax = plt.subplots()
        self.fig2, self.ax2 = plt.subplots()
        self.fig3, self.ax3 = plt.subplots(2,1)
        self.image = self.cube1[:,:,0]
        self.view_image = self.ax.imshow(self.image, cmap = 'gray',vmax=1.0)
        self.view_image2 = None
        self.fig.canvas.draw_idle()
        self.ax.set_title(self.i)
        self.ax.add_patch(self.rect)
        self.ax2.set_aspect(0.1)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.coords_array = []
        self.coords_index = []
        self.coords_dict = {}
        self.x_coords_all = []
        self.i1 = None
        self.i2 = None
        self.y_start = None 
        self.coordinates = None 
        self.y_coords_0 = None
        self.coords_array_new = None 

    def keyPressEvent(self, event):
        if event.key == 'left':
            self.prev()
        elif event.key == 'right':
            self.next()
        elif event.key == 'p':
            self.print_coords()
        elif event.key == 't':
            self.t_h_plot()
        elif event.key == 'u':
            self.save_coords()
        elif event.key == 'c':
            self.comparison()
        elif event.key == 'w':
            self.save_session()
        elif event.key == 'x':
            self.load_coords()

    def print_coords(self):
        x_vals_all = [*set(self.x_coords_all)]
        x_range = np.arange(min(x_vals_all),max(x_vals_all)+1,1)
        keys = list(self.coords_dict.keys())
        print(keys)
        ints = []
        self.n_y = len(x_range)
        self.sheet_cube = np.zeros((self.n_y, self.n))
        self.coordinates = np.zeros((self.n_y,self.n))
        for i in range(len(keys)):
            key = keys[i][:keys[i].find('_')]
            key = int(key)
            ints.append(key)
        ints_no_dups = [*set(ints)]
        ints_no_dups = sorted(ints_no_dups)
        print(ints_no_dups)
        for j in ints_no_dups:
            x_vals = self.coords_dict[str(j)+'_x']
            y_vals = self.coords_dict[str(j)+'_y']
            z = np.polyfit(x_vals, y_vals, 3)
            f = np.poly1d(z)
            x_start = min(x_vals_all)
            x_end = max(x_vals_all)
            y_new = f(x_range)
            if j == min(ints_no_dups):
                self.y_coords_0 = y_new
            else:
                pass
            for k in range(len(x_range)):
                print('k='+str(k))
                x0 = int(x_range[k])
                y0 = int(y_new[k])
                if k == max(x_range):
                    pix_val = ((self.cube1[y0-1,x0,j]/2)+self.cube1[y0,x0,j])/2
                elif k == min(x_range):
                    pix_val = ((self.cube1[y0+1,x0,j]/2)+self.cube1[y0,x0,j])/2
                else:
                    pix_val = ((self.cube1[y0-1,x0,j]/2)+(self.cube1[y0+1,x0,j]/2)+self.cube1[y0,x0,j])/3
                self.sheet_cube[k,j] = pix_val
                self.coordinates[k,j] = y0
        for x in range(self.n):
            if x < min(ints_no_dups):
                for y in range(len(x_range)):
                    x0 = int(x_range[y])
                    y0 = int(self.coordinates[y,x])
                    pix_val = self.cube1[y0,x0,x]
                    self.sheet_cube[y,x] = pix_val 
                    self.coordinates[y,x] = y0
            elif x in ints_no_dups:
                if x == max(ints_no_dups):
                    pass 
                else:
                    self.i1 = int(x)
                    i2_index = int(ints_no_dups.index(x)+1)
                    self.i2 = ints_no_dups[i2_index]
            else:
                for y in range(len(x_range)):
                    x0 = int(x_range[y])
                    i_diff = self.i2 - self.i1
                    y1 = self.coordinates[y,self.i1]
                    y2 = self.coordinates[y,self.i2]
                    print('y='+str(y))
                    print('i1='+str(self.i1))
                    print('i2='+str(self.i2))
                    print('y1='+str(y1))
                    print('y2='+str(y2))
                    y_diff = y2 - y1
                    ints_0 = y_diff/i_diff
                    y_prev = self.coordinates[y,x-1]
                    print('y_prev='+str(y_prev))
                    if y_diff == 0:
                        y0 = int(y_prev)
                    else: 
                        y0 = int(round(y_prev + ints_0))
                    pix_val = ((self.cube1[y0-1,x0,x]/2)+(self.cube1[y0+1,x0,x]/2)+self.cube1[y0,x0,x])/3
                    self.sheet_cube[y,x] = pix_val 
                    self.coordinates[y,x] = y0
        self.ax2.imshow(np.flip(self.sheet_cube,axis=0), cmap = 'magma',vmax = 0.5)
        self.ax2.set_aspect('auto')
        self.ax2.invert_yaxis()
        self.fig2.canvas.draw()

    def save_coords(self):
        for i in self.x_vals:
            self.x_coords_all.append(i)
        self.coords_dict[str(self.i)+'_x'] = self.x_vals
        self.coords_dict[str(self.i)+'_y'] = self.y_vals
        self.ax.plot(self.x_vals, self.y_vals)
        self.fig.canvas.draw()
        self.x_vals = []
        self.y_vals = []

    def save_session(self):
        timestr = time.strftime("%Y%m%d%H%M%S")
        file_name = 't_h_plot_coords_'+str(timestr)
        h5_file = h5py.File(file_name,'w')
        h5_file.create_dataset('x_coords_all',data = self.x_coords_all)
        dict_group = h5_file.create_group('coords_dict')
        for k, v in self.coords_dict.items():
            dict_group[k] = v 
        h5_file.close()

    def load_coords(self):
        self.x_coords_all = self.coords_from_file
        self.coords_dict = {}
        dict_group_load = self.coords_file['coords_dict']
        dict_group_keys = dict_group_load.keys()
        for k in dict_group_keys:
            self.coords_dict[k] = dict_group_load[k][:]
        

    #def t_h_plot(self):
    #    self.n_y = len(self.x_new)
    #    self.sheet_cube = np.zeros((self.n_y, self.n))
    #    for i in range(self.n):
    #        if i in self.coords:
    #            y_i = self.coords.index(i)
    #            y_vals = self.coords_array[y_i]
    #            for j in self.
                #x = self.x_new[j]
                #if i < 101:
                    #y = round(self.y_new[j])
                #else:
                    #y = round(self.y_new[j] - round((self.i-100)/15.0))
                #if j == len(self.x_new) or j == len(self.x_new)-1:
                    #self.pix_val = self.cube1[y,x,i]
                #else:
                    ##if self.cube1[y,x,i] > 0.5:
                        ##self.pix_val = (self.cube2[y-1,x,i] + self.cube2[y,x,i] + self.cube2[y+1,x,i])/3
                        ##self.sheet_cube[j,i] = self.pix_val
                    ##else:
                    #self.pix_val = (self.cube1[y-8,x,i]+self.cube1[y-7,x,i]+self.cube1[y-6,x,i]+self.cube1[y-5,x,i]+self.cube1[y-4,x,i]+self.cube1[y-3,x,i]+self.cube1[y-2,x,i]+self.cube1[y-1,x,i]+self.cube1[y,x,i]+self.cube1[y+1,x,i]+self.cube1[y+2,x,i]+self.cube1[y+3,x,i]+self.cube1[y+4,x,i]+self.cube1[y+5,x,i]+self.cube1[y+6,x,i]+self.cube1[y+7,x,i]+self.cube1[y+8,x,i])/17.0
                    #self.sheet_cube[j,i] = self.pix_val
    #    self.ax2.imshow(np.flip(self.sheet_cube,axis=0), cmap = 'magma',vmax = 0.5)
    #    self.ax2.set_aspect('auto')
    #    self.ax2.invert_yaxis()
    #    self.fig2.canvas.draw()

    def comparison(self):
        base = self.sheet_cube
        average_by_row = self.sheet_cube
        self.ax3[0,0].imshow(base,cmap = 'magma',vmax=0.1)
        self.ax3[1,0].imshow(average_by_row, cmap='magma')
        self.ax3[0,0].title.set_text('Original')
        self.ax3[1,0].title.set_text('New')
        self.ax3[0,0].invert_yaxis()
        self.ax3[1,0].invert_yaxis()
        self.fig3.canvas.draw()


    def t_h_indicator(self):
        if self.line1 == None and self.line2 == None:
            pass
        elif self.line1 == None and self.line2 != None:
            self.line2.remove()
        else:
            self.line1.remove()
            self.line2.remove()
        if self.i == 0:
            self.line2 = self.ax2.axvline(x=self.i+1)
        elif self.i == self.n - 1:
            self.line1 = self.ax2.axvline(x=self.i-1)
        else:
            self.line1 = self.ax2.axvline(x=self.i-1)
            self.line2 = self.ax2.axvline(x=self.i+1)
        self.fig2.canvas.draw()

    def prev(self):
        if self.i == 0:
            self.i = 0
        else:
            self.i = self.i - 1
        self.create_image()
        self.t_h_indicator()

    def next(self):
        if self.i == self.n - 1:
            self.i = self.n - 1
        else:
            self.i = self.i + 1
        self.create_image()
        self.t_h_indicator()

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.x0 = round(self.x0)
        self.y0 = round(self.y0)
        self.x_vals.append(self.x0)
        self.y_vals.append(self.y0)
#        if self.points == None:
#            pass
#        else:
#            for i in self.points:
#                i.remove()
#            self.x_vals = []
#            self.y_vals = []
#            self.fig.canvas.draw()
#        self.x0 = event.xdata
#        self.y0 = event.ydata
    
#    def on_release(self, event):
#        self.x1 = event.xdata
#        self.y1 = event.ydata
#        self.p_start = [round(self.x0),round(self.y0)]
#        self.p_end = [round(self.x1),round(self.y1)]
#        n_cols = round(self.x1) - round(self.x0) + 1
#        self.y = np.round(self.y0)
#        for j in np.arange(1,n_cols):
#            x = round(self.x0) + j
#            a = self.cube[x,round(self.y)-2:round(self.y)+3,self.i]
#            print(self.y)
#            max_y = np.argmax(a) - 2
#            print(max_y)
#            y_new = self.y + max_y
#            self.x_vals.append(x)
#            self.y_vals.append(y_new)
#            self.y = y_new
#        print(self.y_vals)
#        self.points = self.ax.plot(self.x_vals,self.y_vals, color='red')
#        self.fig.canvas.draw()
        

            

    def create_image(self):
        if self.ax.lines:
            self.ax.lines.remove(self.ax.lines[0])
        else:
            pass
        self.ax.set_title(self.i)
        self.image = self.cube1[:,:,self.i]
        self.view_image.set_data(self.image)
        self.fig.canvas.draw_idle()


def save_cutout(files, position, size):
    for i in files:
        hdu = fits.open(i, ignore_missing_simple=True)[1]
        wcs = WCS(hdu.header)
        cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)
        hdu.data = cutout.data  
        hdu.header.update(cutout.wcs.to_header())
        filename = 'cropped_'+str(i)
        hdu.writeto(filename, output_verify='ignore')
    print('Files Cropped_quarter_')


class select_structure_file():
    def __init__(self, cube1):
        self.cube1 = cube1[550:675,0:500,145:270]
        #self.cube2 = cube2[700:850, 970:1280,:]
        self.n = self.cube1.shape[2]
        self.n_2 = None
        self.i = 0
        self.coords_file = h5py.File('t_h_plot_coords_20221104142846','r')
        self.coords_from_file = self.coords_file['x_coords_all']
        self.ints_from_file = self.coords_file['ints_no_dups']
        self.rect = Rectangle((0,0), 1, 1,fill=None)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.line1 = None
        self.line2 = None
        self.line_1 = None
        self.line_2 = None
        self.max_ints = []
        self.sheet_cube = None
        self.sheet2 = None
        self.n_y = None
        self.fit = None
        self.points = None
        self.pix_val = None
        self.y = 0
        self.z = None
        self.f = None
        self.x_vals = []
        self.y_vals = []
        self.coords = []
        self.fig, self.ax = plt.subplots()
        self.fig2, self.ax2 = plt.subplots()
        self.fig3, self.ax3 = plt.subplots(2,1)
        self.image = self.cube1[:,:,0]
        self.view_image = self.ax.imshow(self.image, cmap = 'gray',vmax=1.0)
        self.view_image2 = None
        self.fig.canvas.draw_idle()
        self.ax.set_title(self.i)
        self.ax.add_patch(self.rect)
        self.ax2.set_aspect(0.1)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.coords_array = []
        self.coords_index = []
        self.coords_dict = {}
        self.x_coords_all = []
        self.i1 = None
        self.i2 = None
        self.y_start = None 
        self.coordinates = None 
        self.y_coords_0 = None
        self.coords_array_new = None 
        self.ints_no_dups

    def keyPressEvent(self, event):
        if event.key == 'left':
            self.prev()
        elif event.key == 'right':
            self.next()
        elif event.key == 'p':
            self.print_coords()
        elif event.key == 't':
            self.t_h_plot()
        elif event.key == 'u':
            self.save_coords()
        elif event.key == 'c':
            self.comparison()
        elif event.key == 'w':
            self.save_session()
        elif event.key == 'x':
            self.load_coords()

    def print_coords(self):
        x_vals_all = [*set(self.x_coords_all)]
        x_range = np.arange(min(x_vals_all),max(x_vals_all)+1,1)
        keys = list(self.coords_dict.keys())
        print(keys)
        ints = []
        self.n_y = len(x_range)
        self.sheet_cube = np.zeros((self.n_y, self.n))
        self.coordinates = np.zeros((self.n_y,self.n))
        for i in range(len(keys)):
            key = keys[i][:keys[i].find('_')]
            key = int(key)
            ints.append(key)
        if self.ints_no_dups == None:
            ints_no_dups = [*set(ints)]
            ints_no_dups = sorted(ints_no_dups)
            self.ints_no_dups = ints_no_dups 
        else:
            ints_no_dups = self.ints_no_dups
        print(ints_no_dups)
        for j in ints_no_dups:
            x_vals = self.coords_dict[str(j)+'_x']
            y_vals = self.coords_dict[str(j)+'_y']
            z = np.polyfit(x_vals, y_vals, 3)
            f = np.poly1d(z)
            x_start = min(x_vals_all)
            x_end = max(x_vals_all)
            y_new = f(x_range)
            if j == min(ints_no_dups):
                self.y_coords_0 = y_new
            else:
                pass
            for k in range(len(x_range)):
                print('k='+str(k))
                x0 = int(x_range[k])
                y0 = int(y_new[k])
                if k == max(x_range):
                    pix_val = ((self.cube1[y0-1,x0,j]/2)+self.cube1[y0,x0,j])/2
                elif k == min(x_range):
                    pix_val = ((self.cube1[y0+1,x0,j]/2)+self.cube1[y0,x0,j])/2
                else:
                    pix_val = ((self.cube1[y0-1,x0,j]/2)+(self.cube1[y0+1,x0,j]/2)+self.cube1[y0,x0,j])/3
                self.sheet_cube[k,j] = pix_val
                self.coordinates[k,j] = y0
        for x in range(self.n):
            if x < min(ints_no_dups):
                for y in range(len(x_range)):
                    x0 = int(x_range[y])
                    y0 = int(self.coordinates[y,x])
                    pix_val = self.cube1[y0,x0,x]
                    self.sheet_cube[y,x] = pix_val 
                    self.coordinates[y,x] = y0
            elif x in ints_no_dups:
                if x == max(ints_no_dups):
                    pass 
                else:
                    self.i1 = int(x)
                    i2_index = int(ints_no_dups.index(x)+1)
                    self.i2 = ints_no_dups[i2_index]
            else:
                for y in range(len(x_range)):
                    x0 = int(x_range[y])
                    i_diff = self.i2 - self.i1
                    y1 = self.coordinates[y,self.i1]
                    y2 = self.coordinates[y,self.i2]
                    print('y='+str(y))
                    print('i1='+str(self.i1))
                    print('i2='+str(self.i2))
                    print('y1='+str(y1))
                    print('y2='+str(y2))
                    y_diff = y2 - y1
                    ints_0 = y_diff/i_diff
                    y_prev = self.coordinates[y,x-1]
                    print('y_prev='+str(y_prev))
                    if y_diff == 0:
                        y0 = int(y_prev)
                    else: 
                        y0 = int(round(y_prev + ints_0))
                    pix_val = ((self.cube1[y0-1,x0,x]/2)+(self.cube1[y0+1,x0,x]/2)+self.cube1[y0,x0,x])/3
                    self.sheet_cube[y,x] = pix_val 
                    self.coordinates[y,x] = y0
        self.ax2.imshow(np.flip(self.sheet_cube,axis=0), cmap = 'magma',vmax = 0.5)
        self.ax2.set_aspect('auto')
        self.ax2.invert_yaxis()
        self.fig2.canvas.draw()

    def save_coords(self):
        for i in self.x_vals:
            self.x_coords_all.append(i)
        self.coords_dict[str(self.i)+'_x'] = self.x_vals
        self.coords_dict[str(self.i)+'_y'] = self.y_vals
        self.ax.plot(self.x_vals, self.y_vals)
        self.fig.canvas.draw()
        self.x_vals = []
        self.y_vals = []

    def save_session(self):
        timestr = time.strftime("%Y%m%d%H%M%S")
        file_name = 't_h_plot_coords_'+str(timestr)
        h5_file = h5py.File(file_name,'w')
        h5_file.create_dataset('x_coords_all',data = self.x_coords_all)
        h5_file.create_dataset('ints_no_dups',data = self.ints_no_dups)
        h5_file.close()

    def load_coords(self):
        self.x_coords_all = self.coords_from_file

    #def t_h_plot(self):
    #    self.n_y = len(self.x_new)
    #    self.sheet_cube = np.zeros((self.n_y, self.n))
    #    for i in range(self.n):
    #        if i in self.coords:
    #            y_i = self.coords.index(i)
    #            y_vals = self.coords_array[y_i]
    #            for j in self.
                #x = self.x_new[j]
                #if i < 101:
                    #y = round(self.y_new[j])
                #else:
                    #y = round(self.y_new[j] - round((self.i-100)/15.0))
                #if j == len(self.x_new) or j == len(self.x_new)-1:
                    #self.pix_val = self.cube1[y,x,i]
                #else:
                    ##if self.cube1[y,x,i] > 0.5:
                        ##self.pix_val = (self.cube2[y-1,x,i] + self.cube2[y,x,i] + self.cube2[y+1,x,i])/3
                        ##self.sheet_cube[j,i] = self.pix_val
                    ##else:
                    #self.pix_val = (self.cube1[y-8,x,i]+self.cube1[y-7,x,i]+self.cube1[y-6,x,i]+self.cube1[y-5,x,i]+self.cube1[y-4,x,i]+self.cube1[y-3,x,i]+self.cube1[y-2,x,i]+self.cube1[y-1,x,i]+self.cube1[y,x,i]+self.cube1[y+1,x,i]+self.cube1[y+2,x,i]+self.cube1[y+3,x,i]+self.cube1[y+4,x,i]+self.cube1[y+5,x,i]+self.cube1[y+6,x,i]+self.cube1[y+7,x,i]+self.cube1[y+8,x,i])/17.0
                    #self.sheet_cube[j,i] = self.pix_val
    #    self.ax2.imshow(np.flip(self.sheet_cube,axis=0), cmap = 'magma',vmax = 0.5)
    #    self.ax2.set_aspect('auto')
    #    self.ax2.invert_yaxis()
    #    self.fig2.canvas.draw()

    def comparison(self):
        base = self.sheet_cube
        average_by_row = self.sheet_cube
        self.ax3[0,0].imshow(base,cmap = 'magma',vmax=0.1)
        self.ax3[1,0].imshow(average_by_row, cmap='magma')
        self.ax3[0,0].title.set_text('Original')
        self.ax3[1,0].title.set_text('New')
        self.ax3[0,0].invert_yaxis()
        self.ax3[1,0].invert_yaxis()
        self.fig3.canvas.draw()


    def t_h_indicator(self):
        if self.line1 == None and self.line2 == None:
            pass
        elif self.line1 == None and self.line2 != None:
            self.line2.remove()
        else:
            self.line1.remove()
            self.line2.remove()
        if self.i == 0:
            self.line2 = self.ax2.axvline(x=self.i+1)
        elif self.i == self.n - 1:
            self.line1 = self.ax2.axvline(x=self.i-1)
        else:
            self.line1 = self.ax2.axvline(x=self.i-1)
            self.line2 = self.ax2.axvline(x=self.i+1)
        self.fig2.canvas.draw()

    def prev(self):
        if self.i == 0:
            self.i = 0
        else:
            self.i = self.i - 1
        self.create_image()
        self.t_h_indicator()

    def next(self):
        if self.i == self.n - 1:
            self.i = self.n - 1
        else:
            self.i = self.i + 1
        self.create_image()
        self.t_h_indicator()

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.x0 = round(self.x0)
        self.y0 = round(self.y0)
        self.x_vals.append(self.x0)
        self.y_vals.append(self.y0)
#        if self.points == None:
#            pass
#        else:
#            for i in self.points:
#                i.remove()
#            self.x_vals = []
#            self.y_vals = []
#            self.fig.canvas.draw()
#        self.x0 = event.xdata
#        self.y0 = event.ydata
    
#    def on_release(self, event):
#        self.x1 = event.xdata
#        self.y1 = event.ydata
#        self.p_start = [round(self.x0),round(self.y0)]
#        self.p_end = [round(self.x1),round(self.y1)]
#        n_cols = round(self.x1) - round(self.x0) + 1
#        self.y = np.round(self.y0)
#        for j in np.arange(1,n_cols):
#            x = round(self.x0) + j
#            a = self.cube[x,round(self.y)-2:round(self.y)+3,self.i]
#            print(self.y)
#            max_y = np.argmax(a) - 2
#            print(max_y)
#            y_new = self.y + max_y
#            self.x_vals.append(x)
#            self.y_vals.append(y_new)
#            self.y = y_new
#        print(self.y_vals)
#        self.points = self.ax.plot(self.x_vals,self.y_vals, color='red')
#        self.fig.canvas.draw()
        

            

    def create_image(self):
        if self.ax.lines:
            self.ax.lines.remove(self.ax.lines[0])
        else:
            pass
        self.ax.set_title(self.i)
        self.image = self.cube1[:,:,self.i]
        self.view_image.set_data(self.image)
        self.fig.canvas.draw_idle()

