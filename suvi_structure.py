import numpy as np
import matplotlib.pyplot as plt
import savitzkygolay

# As set up now, this requires an array of coordinates to be passed into the class, 
# along with the two data sets (here cube1 is the long exposures and cube 2 is the 
# short exposures). You can click the image generated from the input data to create
# a list of coordinates, and then either hit the 'u' key to save coordinates to a 
# .npy file or hit the 'p' key to create a polyline from the stored coordinates.
# Hit the 't' key to generate a time-height plot, and the 'c' key to create a 
# comparison set of plots with the noise-reduction techniques used below.
# I'm going to modify the class to not require a coordinates array by default.

class select_structure():
    def __init__(self, cube1, cube2,array):
        self.cube1 = cube1[700:850,970:1280,:]
        self.cube2 = cube2[700:850, 970:1280,:]
        self.n = self.cube1.shape[2]
        self.n_2 = None
        self.i = 0
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.line1 = None
        self.line2 = None
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
        self.x_vals = array[0]
        self.y_vals = array[1]
        self.coords = []
        self.fig, self.ax = plt.subplots()
        self.fig2, self.ax2 = plt.subplots()
        self.fig3, self.ax3 = plt.subplots(2,2)
        self.image = self.cube1[:,:,0]
        self.view_image = self.ax.imshow(self.image, cmap = 'goes-rsuvi131')
        self.view_image2 = None
        self.fig.canvas.draw_idle()
        self.ax.set_title(self.i)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.on_press)


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

# 
    def print_coords(self):
        self.z = np.polyfit(self.x_vals, self.y_vals, 3)
        self.f = np.poly1d(self.z)
        self.x_start = round(self.x_vals[0])
        self.x_end = round(self.x_vals[-1])
        self.x_new = np.arange(self.x_start,self.x_end+1,1)
        self.y_new = self.f(self.x_new)
        self.ax.plot(self.x_new,self.y_new)
        self.fig.canvas.draw()

    def save_coords(self):
        np.save(file = 'coords.npy',arr = [self.x_new,self.y_new])

    def t_h_plot(self):
        self.n_y = len(self.x_new)
        self.sheet_cube = np.zeros((self.n_y, self.n))
        for i in range(self.n):
            for j in range(len(self.x_new)):
                x = self.x_new[j]
                y = round(self.y_new[j])
                if j == len(self.x_new) or j == len(self.x_new)-1:
                    self.pix_val = self.cube1[y,x,i]
                else:
                    if self.cube1[y,x,i] > 0.5:
                        self.pix_val = (self.cube2[y-1,x,i] + self.cube2[y,x,i] + self.cube2[y+1,x,i])/3
                        self.sheet_cube[j,i] = self.pix_val
                    else:
                        self.pix_val = (self.cube1[y-1,x,i]+self.cube1[y,x,i]+self.cube1[y+1,x,i])/3
                        self.sheet_cube[j,i] = self.pix_val
        self.ax2.imshow(self.sheet_cube, cmap = 'magma',vmax = 0.1)
        self.ax2.invert_yaxis()
        self.fig2.canvas.draw()

    def comparison(self):
        base = self.sheet_cube
        average_by_row = self.sheet_cube
        s_g_rows = self.sheet_cube
        for k in range(self.n_y):
            average = np.mean(self.sheet_cube[k,:])
            for l in range(self.n):
                pix_val = self.sheet_cube[k,l] - average
                average_by_row[k,l] = pix_val
        s_g = savitzkygolay.filter2D(self.sheet_cube, 9, 3)
        #for i in range(self.n_y):
        #    row = savitzkygolay.filter1D(self.sheet_cube[i,:],9,3)
        #    s_g_rows[i,:] = row
        self.ax3[0,0].imshow(base,cmap = 'magma',vmax=0.1)
        self.ax3[0,1].imshow(average_by_row, cmap='magma',vmax=0.1)
        self.ax3[1,0].imshow(s_g, cmap='magma',vmax=0.1)
        #self.ax3[1,1].imshow(s_g_rows, cmap='magma',vmax=0.1)
        self.ax3[0,0].title.set_text('Original')
        self.ax3[0,1].title.set_text('Averaged by row')
        self.ax3[1,0].title.set_text('Savitzky-Golay')
        self.ax3[1,1].title.set_text('SG Rows')
        self.ax3[0,0].invert_yaxis()
        self.ax3[0,1].invert_yaxis()
        self.ax3[1,0].invert_yaxis()
        self.ax3[1,1].invert_yaxis()
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
        self.x_vals.append(self.x0)
        self.y_vals.append(self.y0)

    def create_image(self):
        self.ax.set_title(self.i)
        self.image = np.flip(self.cube1[:,:,self.i],axis=0)
        self.view_image.set_data(self.image)
        self.fig.canvas.draw_idle()


