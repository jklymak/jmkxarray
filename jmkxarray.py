import xarray as xr
import numpy as np
import scipy
import logging

_log = logging.getLogger(__name__)

def gappy_interp(ds, *, dim='', xgrid=None, maxgap=np.Inf):
    """

    OBSOLETE: use `ds.interpolate_na(maxgap=)
    interp ds on xgrid along dimension dim.  However don't
    interpolate across gaps larger than maxgap.



    Parameters
    ----------
    ds : xarray.DataSet or xarray.DataFrame
        data frame.  Must have *dim* as a dimension
    dim : string
        dimension to interpolate on.
    xgrid : array
        grid to interpolate to in dimension *dim*
    maxgap : float
        maximum size of gap in *dim* to interpolate across

    Returns
    -------
    DataSet or DataArray with dim=xgrid.

    Examples
    --------
    Note in the below that the gap from x=1 to 2.1 is _not_ interpolated over.

    >>> da = xr.DataArray(
        ...     data=[[1, 2, 3, 4, 5, 6]],
        ...     dims=("x"),
        ...     coords={"x": [0, 0.5, 1, 2.1, 2.5, 3.4]},
        ... )
    >>> dn = gappy_interp(da, dim='x', xgrid=np.arange(0, 3.1, 0.2), maxgap=1.0)
    >>> dn
    <xarray.DataArray (x: 16)>
    array([1.        , 1.4       , 1.8       , 2.2       , 2.6       ,
           3.        ,        nan,        nan,        nan,        nan,
           nan, 4.25      , 4.75      , 5.11111111, 5.33333333, 5.55555556])
    Coordinates:
    * x        (x) float64 0.0 0.2 0.4 0.6 0.8 1.0 1.2 ... 2.0 2.2 2.4 2.6 2.8 3.0
    """

    x0 = ds[dim].values
    print('xgrid:', xgrid)
    print('dim', dim)
    ds = ds.interp(**{dim:xgrid})
    dx = np.diff(x0)
    bad = (dx > maxgap).nonzero()[0]
    print("Bad", bad)
    mask = np.isfinite(xgrid)
    for b in bad:
        print("B", b, x0[b], x0[b+1])
        mask[(xgrid > x0[b]) & (xgrid < x0[b+1])] = False
    mask = xr.DataArray(mask, dims=(dim))
    return ds.where(mask)


def depth_to_iso(ds, pden='pden', depths='depths', xdim='along',
                 pden0=None, isodepths=None):
    """
    Map fields in this dataset to potential density coordinates.

    Parameters
    -----------

    ds : xarray Dataset
        Dataset that at least has *pden* and *depths* fields to map to

    pden : str (default 'pden')
        Name of Array in *ds* that has the density information.

    depths : str (default 'depths')
        Name of depth coordinate that has depth information.

    xdim : str (default 'along')
        Name of x coordinate of the data set

    pden0 : array-like (default None)
        Array of mean densities to interpolate onto.  If not provided, the
        data set mean is used.

    isodepths : array-like (default None)
        Mean depths of the isopycnals defined in *pden0*.
    """

    if pden0 is None:
        pden0 = ds[pden].mean(dim=xdim)
        isodepths = ds[depths].where(pden0 > 0).dropna(dim=depths)
        pden0 = pden0.dropna(dim=depths)

    M = pden0.shape[0]
    N = ds[xdim].shape[0]
    print(M, N)

    dsiso = xr.Dataset(coords={'isodepths':('isodepths', isodepths.values),
                               xdim: (xdim, ds[xdim].values)})
    print(dsiso)
    print(len(isodepths.values))
    for td in ds.variables:
        _log.info('todo', td, ds[td].shape)
        print(td, ds[td].shape)
        if td == xdim:
            pass
        elif td == depths:
            pass
        elif ds[td].shape == ds[xdim].shape:
            dsiso[td] = (xdim, ds[td].values)
        elif ds[td].shape == ds[pden].shape:
            dsiso[td] = (('isodepths', xdim), np.zeros((M, N)))
            for i in range(N):
                dsiso[td][:, i] = np.interp(pden0, ds[pden][:, i], ds[td].values[:, i])
        else:
            try:
                dsiso[td] = ds[td]
            except ValueError:
                pass

    return dsiso


def get_n_blocks(nfft, N, noverlap_min=None):
    if noverlap_min is None:
        noverlap_min = nfft / 2
    for m in range(2, 2000):
        noverlap = np.ceil(- (N - m * nfft) / (m - 1))
        if noverlap > 0 and noverlap >= noverlap_min:
            total = m * nfft - (m-1) * noverlap
            if total == nfft:
                noverlap = 0
            return int(noverlap), int(total), m


def power_spec(da, nfft=256, xdim='along', ydim='depths', xunits='km',
               dataunits='kg/m^3'):

    """
    Calculate power spectra segments of a DataArray and return as a Dataset.

    Parameters
    ----------

    da : DataArray
        data array to calculate spectra over.  Spectra are calculated in the
        second dimension, usually something like "x" or "time".

    nfft : integer
        length of FFT blocks

    xdim : string
        dimension in DataArray to calculate spectra along.

    ydim : string
        dimension in DataArray over which we iterate to calculate the spectra.

    xunits : string
        units of the xdimension

    dataunits : string
        units of the data that is being analyzed

    Returns
    -------

    ps : DataSet
        Data set with coordinates (blocks, depths, kx), if ydim is "depths".  Note that you
        may want to rename the coordinates if kx is a frequency.  Note that kx has units of
        cycles / km, *not* rad / km.  The spectrum has units of V^2 / cpkm, if "V" are the
        units of the data in *da*, and is normalized so that the integral of S(kx) dkx is
        equal to the variance of the signal in *da*.

    """
    N = da[xdim].shape[0]
    window = np.hanning(nfft)
    wnorm = (window * window).sum()
    noverlap, total, nblocks = get_n_blocks(nfft, N, noverlap_min=nfft/2)
    _log.info(noverlap, nblocks)
    dx = da[xdim].diff(dim=xdim).median().values
    kx = np.arange(0, 1/2-1e-5, 1/nfft) / dx
    ps =  xr.Dataset(coords={'blocks': np.arange(nblocks), 'depths': da[ydim].values, 'kx': kx })
    ps['kx'].attrs = {'description': 'wavenumber or frequency',
                      'units': f'cycles per {xunits}'}
    ps['spectrum'] = (('blocks', 'depths', 'kx'), np.zeros((nblocks, da[ydim].shape[0], len(kx))))
    ps['spectrum'].attrs = {'description': 'un-averaged power spectra from overlaping blocks',
                            'units': f'({dataunits})^2 / cp{xunits}',
                            'source': f'{da.name}'}
    ps['along_start'] = (('blocks'), np.zeros(nblocks))
    ps['along_stop'] = (('blocks'), np.zeros(nblocks))
    ps['along_start'].attrs = {'description' : f'start of the block in {xdim}'}
    ps['along_stop'].attrs = {'description' : f'start of the block in {xdim}'}
    ps['blocks'].attrs = {'description' : 'block index for the spectral estimates'}

    start = 0
    for nn in range(nblocks):
        stop = start + nfft
        _log.info(start, stop, da.isel({xdim: slice(start, stop)}).shape)
        xx = da.isel({xdim: slice(start, stop)})
        xx = xx - xx.mean(dim=xdim)
        ff = np.fft.fft(xx * window, axis=1)
        pp = ff[:, :int(nfft/2)]*np.conj(ff[:, :int(nfft/2)]) * 2 * dx / wnorm
        ps['spectrum'][nn, :, : ] = np.real(pp)
        ps['along_start'][nn] = da[xdim].isel({xdim:start})
        ps['along_stop'][nn] = da[xdim].isel({xdim: stop-1})
        start = stop - noverlap
    ps.attrs = {'nblocks': nblocks, 'noverlap': noverlap, 'nfft': nfft}
    return ps


def multi_psd(da, minnfft=64, xdim='along', ydim='depths', xunits='km',
              dataunits='kg/m^3'):

    """
    Calculate power spectra segments of a DataArray with different resolutions,
    and return a spectra.  Interpolate over gaps that are a bit larger than the resolution
    changes

    nfft: size of fft blocks to try.
        Maxgap for the current block will depend on the size of the next block.
    """
    N = len(da[xdim])
    # minnfft = 64
    nffts = [2*int(np.floor(N/3/2)*2)]
    while np.floor(nffts[-1]/6) > minnfft:
        nffts += [np.floor(nffts[-1]/6/2)*2]

    # nffts = [2*int(np.floor(N/3)), 2*int(np.floor(N/18)), 2*int(np.floor(N/56))]

    maxgaps = nffts[1:]
    maxgaps += [2]
    maxgaps = maxgaps[::-1]
    # get the maxgap to interpolate over for each fft attempt
    spec = []
    kx = []
    for nn, nfft in enumerate(nffts[::-1]):
        dd = da.interpolate_na(dim=xdim, max_gap=maxgaps[nn])

        p = power_spec(dd, nfft=int(nfft), xdim=xdim, ydim=ydim,
                        xunits=xunits, dataunits=dataunits).mean(dim='blocks')
        if nn == 0:
            kmax = None
            spec = p.spectrum.values[:, 1:]
            kx = p.kx.values[1:]
        else:
            kmax = kx[0] - dkx / 2
            spec = np.concatenate((p.spectrum.sel(kx=slice(None, kmax)).values[:, 1:], spec), axis=1)
            kx = np.concatenate((p.kx.sel(kx=slice(None, kmax)).values[1:], kx))
        dkx = p.kx.diff(dim='kx').median(dim='kx')
    dout = xr.Dataset(coords={'depths': da[ydim].values, 'kx': kx })
    dout['kx'] = ('kx', kx)
    dout['spectrum'] = (('depths', 'kx'), spec)

    return dout


def whiten(kx, sp):
    return sp*(np.pi*2*kx)**2


