import pathlib

import eispac


if __name__ == '__main__':
    eis_h5_dir = pathlib.Path(snakemake.input[0])
    eis_h5_files = list(eis_h5_dir.glob('*.data.h5'))
    eis_fits_dir = pathlib.Path(snakemake.output[0])
    eis_fits_dir.mkdir(exist_ok=True, parents=True)
    
    # NOTE: I manually selected these by first fitting all of the lines and then
    # going back and looking at which ones looked "good" (i.e. not noisy)
    template_names = [
        'ar_11_188_806.3c.template.h5',
        'ar_14_187_964.1c.template.h5',
        'ar_14_194_396.2c.template.h5',
        'ca_14_193_874.2c.template.h5',
        'ca_15_200_972.2c.template.h5',
        'ca_17_192_858.1c.template.h5',
        'fe_08_185_213.1c.template.h5',
        'fe_08_186_601.1c.template.h5',
        'fe_09_188_497.3c.template.h5',
        'fe_09_197_862.1c.template.h5',
        'fe_10_184_536.1c.template.h5',
        'fe_10_193_715.4c.template.h5',
        'fe_11_180_401.1c.template.h5',
        'fe_11_188_216.2c.template.h5',
        'fe_11_188_299.2c.template.h5',
        'fe_11_192_813.2c.template.h5',
        'fe_12_186_880.1c.template.h5',
        'fe_12_192_394.1c.template.h5',
        'fe_12_195_119.2c.template.h5',
        'fe_12_195_179.2c.template.h5',
        'fe_12_203_720.2c.template.h5',
        'fe_13_201_121.2c.template.h5',
        'fe_13_202_044.1c.template.h5',
        'fe_13_203_826.2c.template.h5',
        'fe_14_264_787.1c.template.h5',
        'fe_14_270_519.2c.template.h5',
        'fe_15_284_160.2c.template.h5',
        'fe_16_262_984.1c.template.h5',
        'fe_24_192_040.1c.template.h5',
        'mg_06_270_394.2c.template.h5',
        's__13_256_686.1c.template.h5',
        'si_07_275_368.3c.template.h5',
        'si_10_258_375.1c.template.h5',
    ]
    template_filenames = [eispac.data.get_fit_template_filepath(tn) for tn in template_names]

    for filename in template_filenames:
        template = eispac.EISFitTemplate.read_template(filename)
        cube = eispac.read_cube(eis_h5_files[0], window=template.central_wave)
        fit_res = eispac.fit_spectra(cube, template, ncpu='max')
        _ = eispac.export_fits(fit_res, save_dir=eis_fits_dir)
