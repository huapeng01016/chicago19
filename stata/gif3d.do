cscript

version 16

python:
import numpy as np
from sfi import Platform
import matplotlib
if Platform.isWindows():
	matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sfi import Data
import imageio as io
import os

stata: sysuse sandstone, clear
D = np.array(Data.get("northing easting depth"))

ax = plt.axes(projection='3d')
plt.xticks(np.arange(60000, 90001, step=10000))
plt.yticks(np.arange(30000, 50001, step=5000))
ax.plot_trisurf(D[:,0], D[:,1], D[:,2], cmap=plt.cm.Spectral, edgecolor='none')

for i in range(0, 360, 10):
	ax.view_init(elev=10., azim=i)
	plt.savefig("sandstone"+str(i)+".png")
	
with io.get_writer('sandstone.gif', mode='I', duration=0.5) as writer:
	for i in range(0, 360, 10):
		image = io.imread("sandstone"+str(i)+".png")
		writer.append_data(image)
writer.close()

end
