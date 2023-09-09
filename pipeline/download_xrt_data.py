"""
Download XRT data

.. note:: For now, these files are all hardcoded as the URLs to the synoptic level 2 composite files.
          This is because getting the level 1 XRT files at the moment is seemingly impossible at the moment.
          At some point, this should be changed to doing an actual Fido query as well as additional processing
          steps to get the files to level 2.
"""
import pathlib
import urllib

import sunpy.map


if __name__ == '__main__':
    file_urls = [
        'http://solar.physics.montana.edu/HINODE/XRT/SCIA/synop_official/2012/09/24/H1000/comp_XRT20120924_100307.8.fits',
        'http://solar.physics.montana.edu/HINODE/XRT/SCIA/synop_official/2012/09/24/H1000/comp_XRT20120924_100346.8.fits',
        'http://solar.physics.montana.edu/HINODE/XRT/SCIA/synop_official/2012/09/24/H1000/comp_XRT20120924_100424.3.fits',
    ]
    file_paths = [pathlib.Path(urllib.parse.urlsplit(url).path) for url in file_urls]
    xrt_maps = sunpy.map.Map(file_urls)
    
    xrt_fits_dir = pathlib.Path(snakemake.output[0])
    xrt_fits_dir.mkdir(exist_ok=True, parents=True)
    for m, fp in zip(xrt_maps, file_paths):
        m.save(xrt_fits_dir / fp.name)
