import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import datetime

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter

def onewidths(nrows=1, ncols=1, vext=0.3, **kwargs):
    """
    """
    wid = 255 / 72
    height = 10 * vext
    fsize = kwargs.pop('figsize', None)
    if fsize is None:
        fsize = (wid, height)
    return plt.subplots(nrows, ncols, figsize=fsize, **kwargs)


def twowidths(nrows=1, ncols=1, vext=0.3, **kwargs):
    wid = 539 / 72
    height = 10 * vext
    fsize = kwargs.pop('figsize', None)
    if fsize is None:
        fsize = (wid, height)
    return plt.subplots(nrows, ncols, figsize=fsize, **kwargs)


def djmkfigure(width,vext):
    """
    djmkfigure(width, vext):
    width is column widths, and vext is fractional 10 page height.
    """
    wid = 3*width+3./8.;
    height = 10*vext;
    plt.rc('figure',figsize=(wid,height),dpi=96)
    plt.rc('font',size=9)
    plt.rc('font',family='sans-serif');
    # rcParams['font.sans-serif'] = ['Verdana']
    plt.rc('axes',labelsize='large')
    leftin = 0.75
    rightin = 0.25
    botin = 0.4
    plt.rc('figure.subplot',left=leftin/wid)
    plt.rc('figure.subplot',right=(1-rightin/wid))
    plt.rc('figure.subplot',bottom=botin/height)

def jmkprint(fname,pyname,dirname='doc',dpi=150,optcopy=False,bbinch=None):
    """
    def jmkprint(fname,pyname)
    def jmkprint(fname,pyname,dirname='doc')
    """
    import os,shutil

    fig = plt.gcf()
    print(fig)
    try:
        os.mkdir(dirname)
    except:
        pass

    if dirname=='doc':
        pwd=os.getcwd()+'/doc/'
    else:
        pwd=dirname+'/'
    fig.savefig(dirname+'/'+fname+'.pdf',dpi=dpi,bbox_inches=bbinch)
    fig.savefig(dirname+'/'+fname+'.png',dpi=dpi,bbox_inches=bbinch)
    fig.savefig(dirname+'/'+fname+'.svg',dpi=dpi,bbox_inches=bbinch)

    str="""\\begin{{figure*}}[htbp]
  \\begin{{center}}
    \\includegraphics[width=\\twowidth]{{{fname}}}
    \\caption{{
      \\tempS{{\\footnotesize {pwd}/{pyname} ;
        {pwd}{fname}.pdf}}
      \\label{{fig:{fname}}} }}
  \\end{{center}}
\\end{{figure*}}""".format(pwd=pwd,pyname=pyname,fname=fname)

    with open(dirname+'/'+fname+'.tex','w') as fout:
        fout.write(str)

    cmd = 'less '+dirname+'/%s.tex | pbcopy' % fname
    os.system(cmd)
    if optcopy:
        shutil.copy(dirname+'/'+fname+'.png',optcopy)


def tsdiagramjmk(salt,temp,cls=[]):
    import numpy as np
    import seawater
    import matplotlib.pyplot as plt


    # Figure out boudaries (mins and maxs)
    smin = salt.min() - (0.01 * salt.min())
    smax = salt.max() + (0.01 * salt.max())
    tmin = temp.min() - (0.1 * temp.max())
    tmax = temp.max() + (0.1 * temp.max())

    # Calculate how many gridcells we need in the x and y dimensions
    xdim = round((smax-smin)/0.1+1,0)
    ydim = round((tmax-tmin)+1,0)


    # Create empty grid of zeros
    dens = np.zeros((ydim,xdim))

    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(1,ydim-1,ydim)+tmin
    si = np.linspace(1,xdim-1,xdim)*0.1+smin

    # Loop to fill in grid with densities
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):
            dens[j,i]=seawater.dens(si[i],ti[j],0)

    # Substract 1000 to convert to sigma-t
    dens = dens - 1000

    # Plot data ***********************************************
    if not(cls==[]):
        CS = plt.contour(si,ti,dens, cls,linestyles='dashed', colors='k')
    else:
        CS = plt.contour(si,ti,dens,linestyles='dashed', colors='k')

    plt.clabel(CS, fontsize=9, inline=1, fmt='%1.2f') # Label every second level
    ax1=gca()
    #    ax1.plot(salt,temp,'or',markersize=4)

    ax1.set_xlabel('S [psu]')
    ax1.set_ylabel('T [C]')
