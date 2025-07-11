"""Take a VNA traces with power sweep."""

import time
import stuelab
from stuelab.devices.VNA_P9372B import VNA_P9372B
import numpy as np

##################
### User Input ###
##################
prefix = 'Group_04'
descr = 'n3_VNA_res2_lin_20'

vna_params = {'traces': ['S21'], 'ifbw': 500, 'points': 1001,
              'start_power': -42, 'end_power': -42, 'step_power': 21,
              'center': [3.609e9, ],
              'span': [6e6, ]}
##################
##################
##################

# Put everything into a try-finally block in order to safely close all devices.
try:
    # defining power list
    power_list = np.linspace(vna_params['start_power'], vna_params['end_power'], vna_params['step_power'])
    # Initialize the devices.
    vna = VNA_P9372B()
    vna.ClearAll()
    vna.AddTraces(vna_params['traces'])
    vna.SetIFBW(vna_params['ifbw'])
    vna.SetPoints(vna_params['points'])
    for i in range(0, len(vna_params['center'])):
        vna.SetCenter(vna_params['center'][i])
        vna.SetSpan(vna_params['span'][i])

        # Start the measurement.
        with stuelab.newfile(prefix, descr+f"_{vna_params['center'][i]/1e9:.2f}GHz", autoindex=True, git_id=False) as dfile:
            for idx_power, power in enumerate(power_list):
                print(f'\rMeasure power {idx_power + 1}/{len(power_list)} ({power}dBm)...', end='', flush=True)

                # Set the power for the VNA
                vna.SetPower(power)
                # Wait for the VNA to settle.
                time.sleep(0.1)
                # Start the measurement.
                time_now = time.time()
                # collect the data
                data = vna.MeasureScreen_pd()
                vna.AutoScaleAll()
                data['Power (dBm)'] = vna.GetPower()
                data['Timestamp (s)'] = time_now
                # Write the data to disk.
                stuelab.saveframe(dfile, data)
                stuelab.metagen.fromarrays(
                    dfile,
                    xarray=data['Frequency (Hz)'],
                    yarray=power_list[:idx_power+1],
                    xtitle='Frequency (Hz)',
                    ytitle='Index',
                    colnames=list(data),
                )
            print()
finally:
    print("Shutting down devices...")
    # Enclose every closing attempt in its own try-except block to ensure that
    # we attempt to close them all even if there is a problem.
    try:
        vna.write('TRIG:SOUR IMM')
        vna.close()
    except: pass
