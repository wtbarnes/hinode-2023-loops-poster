from sunpy.net import Fido, attrs as a
import eispac.net
import astropy.time
import astropy.units as u

t_midpoint = astropy.time.Time(snakemake.config["eis_raster_midpoint"], format='iso', scale='utc')
q = Fido.search(
    a.Time(t_midpoint - 2*u.h, t_midpoint + 2*u.h),
    a.Instrument('EIS'),
    a.Physobs('Intensity'),
    a.Source('Hinode'),
    a.Provider('NRL'),
    a.Level('1'),
    a.eispac.FileType('HDF5 data') | a.eispac.FileType('HDF5 header')
)
files = Fido.fetch(q, path=snakemake.output[0])