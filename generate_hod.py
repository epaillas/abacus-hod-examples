import yaml
import numpy as np
import argparse
from abacusnbody.hod.abacus_hod import AbacusHOD
from pathlib import Path
from cosmoprimo.fiducial import AbacusSummit
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
from astropy.io.fits import Header


def get_rsd_positions(hod_dict):
    """Read positions and velocities from HOD dict
    return real and redshift-space galaxy positions."""
    data = hod_dict['LRG']
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    x = data['x'] + boxsize / 2
    y = data['y'] + boxsize / 2
    z = data['z'] + boxsize / 2
    x_rsd = x + vx / (hubble * az)
    y_rsd = y + vy / (hubble * az)
    z_rsd = z + vz / (hubble * az)
    x_rsd = x_rsd % boxsize
    y_rsd = y_rsd % boxsize
    z_rsd = z_rsd % boxsize
    return x, y, z, x_rsd, y_rsd, z_rsd


def output_mock(mock_dict, newBall, fn, tracer, fmt='fits', overwrite=False):
    """Save HOD catalogue to disk."""
    if fmt == 'npz':
        np.savez(fn, **mock_dict[tracer], 
                 tracer = tracer, 
                 hod = newBall.tracers[tracer])
    elif fmt == 'csv':
        Ncent = mock_dict[tracer]['Ncent']
        mock_dict[tracer].pop('Ncent', None)
        table = Table(mock_dict[tracer], meta = {'Ncent': Ncent, 'Gal_type': tracer, **newBall.tracers[tracer]})
        ascii.write(table, fn, overwrite = overwrite, format = 'ecsv')
    elif fmt == 'fits':
        Ncent = mock_dict[tracer]['Ncent']
        mock_dict[tracer].pop('Ncent', None)
        # create cen column
        cen = np.zeros(len(mock_dict[tracer]['x']))
        cen[:Ncent] += 1
        mock_dict[tracer]['cen'] = cen
        table = Table(mock_dict[tracer])

        header = Header({'Ncent': Ncent, 'Gal_type': tracer, **newBall.tracers[tracer]})
        myfits = fits.BinTableHDU(data = table, header = header)
        fn += '.fits'
        myfits.writeto(fn, overwrite = overwrite)
    else:
        print("file format not recognized, use 'npz', 'fits', or 'csv'. ")


def get_hod(p, param_mapping, param_tracer, data_params, Ball, nthread):
    """Get the dictionary containing HOD information."""
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        if key == 'sigma' and tracer_type == 'LRG':
            Ball.tracers[tracer_type][key] = 10**p[mapping_idx]
        else:
            Ball.tracers[tracer_type][key] = p[mapping_idx]
    Ball.tracers['LRG']['ic'] = 1
    ngal_dict = Ball.compute_ngal(Nthread = nthread)[0]
    N_lrg = ngal_dict['LRG']
    Ball.tracers['LRG']['ic'] = min(1, data_params['tracer_density_mean']['LRG']*Ball.params['Lbox']**3/N_lrg)
    mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = nthread)
    return mock_dict


def setup_hod(config):
    """Set up HOD run using the configuration file."""
    print(f"Processing {config['sim_params']['sim_name']}")
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    fit_params = config['fit_params']    
    newBall = AbacusHOD(sim_params, HOD_params)
    newBall.params['Lbox'] = boxsize
    param_mapping = {}
    param_tracer = {}
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        tracer_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_tracer[key] = tracer_type
    return newBall, param_mapping, param_tracer, data_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)

    args = parser.parse_args()
    start_hod = args.start_hod
    n_hod = args.n_hod
    start_cosmo = args.start_cosmo
    n_cosmo = args.n_cosmo
    start_phase = args.start_phase
    n_phase = args.n_phase

    config_dir = './'
    config_fn = Path(config_dir, 'config.yaml')
    config = yaml.safe_load(open(config_fn))

    boxsize = 2000
    redshift = 1.1
    los = 'z'

    for cosmo in range(start_cosmo, start_cosmo + n_cosmo):
        mock_cosmo = AbacusSummit(cosmo)
        az = 1 / (1 + redshift)
        hubble = 100 * mock_cosmo.efunc(redshift)

        hods_dir = f'./'
        hods_fn = Path(hods_dir, f'hod_parameters.npz')
        hod_params = np.load(hods_fn)['hods']

        for phase in range(start_phase, start_phase + n_phase):
            sim_fn = f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}'
            config['sim_params']['sim_name'] = sim_fn
            newBall, param_mapping, param_tracer, data_params = setup_hod(config)

            for hod in range(start_hod, start_hod + n_hod):
                hod_dict = get_hod(hod_params[hod], param_mapping, param_tracer,
                              data_params, newBall, 256)

                x, y, z, x_rsd, y_rsd, z_rsd = get_rsd_positions(hod_dict)

                xpos = x_rsd if los == 'x' else x
                ypos = y_rsd if los == 'y' else y
                zpos = z_rsd if los == 'z' else z

                data_positions = np.c_[xpos, ypos, zpos]

                # from here on you can do whatever you want with these positions,
                # e.g. do density split or calculate correlation functions

                # or maybe you want to save the catalogue to disk
                output_dir = Path(config['sim_params']['output_dir'],
                                  config['sim_params']['sim_name'],
                                  f'z{redshift:.3f}')

                output_fn = str(output_dir / f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}_hod{hod}')
                output_mock(hod_dict, newBall, output_fn, 'LRG', fmt='fits')
