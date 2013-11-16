from pylab import *
import time

def display_iterate(stack,realigner):
    stack._init_dataset()
    nrows = int(np.ceil(np.sqrt(stack.nslices)))
    fig, plots = subplots(nrows,nrows,facecolor='black')
    fig.subplots_adjust(hspace=0,wspace=0)
    for pp in plots:
        for p in pp:
            p.set_axis_off()
    pfr = 0
    for fr,sl,aff,tt,slice_data in stack.iter_slices():
        if pfr != fr :
            pause(1)
            pfr=fr
        plt = plots[int(np.floor(sl/nrows))][sl%nrows]
        plt.matshow(slice_data,cmap='gray')
        fig.canvas.draw()
        time.sleep(.01)

        
        
        
