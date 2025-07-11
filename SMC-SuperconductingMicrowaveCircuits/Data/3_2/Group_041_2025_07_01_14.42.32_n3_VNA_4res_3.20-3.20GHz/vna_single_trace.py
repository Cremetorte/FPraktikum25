"""Take a VNA trace."""

import time
import stuelab
from stuelab.devices.VNA_P9372B import VNA_P9372B
##################
### User Input ###
##################
prefix = 'Group_04'
descr = 'n3_VNA_4res'

vna_params = {'traces': ['S21'], 'ifbw': 100, 'power': -50, 'points': 1001}
vna_params.update({'start': 3.2e9, 'end': 4.7e9})
#vna_params.update({'center': None, 'span': None})

##################
##################
##################

# Put everything into a try-finally block in order to safely close all devices.
try:
    # Initialize the devices.
    vna = VNA_P9372B()
    vna.ClearAll()
    vna.AddTraces(vna_params['traces'])
    vna.SetIFBW(vna_params['ifbw'])
    vna.SetPower(vna_params['power'])
    vna.SetPoints(vna_params['points'])
    if 'start' in vna_params and 'end' in vna_params:
        vna.SetStart(vna_params['start'])
        vna.SetEnd(vna_params['end'])
        freq = f"_{vna_params['start'] / 1e9:.2f}-{vna_params['start'] / 1e9:.2f}GHz"

    elif 'center' in vna_params and 'span' in vna_params:
        vna.SetCenter(vna_params['center'])
        vna.SetSpan(vna_params['span'])
        freq = f"_{vna_params['center'] / 1e9:.2f}GHz"
    # Start the measurement.
    with stuelab.newfile(prefix, descr+freq, autoindex=True, git_id=False) as dfile:
        # Collect the data.
        time_now = time.time()
        data = vna.MeasureScreen_pd()
        vna.AutoScaleAll()
        data['Power (dBm)'] = vna.GetPower()
        data['Timestamp (s)'] = time_now
        # Write the data to disk.
        stuelab.saveframe(dfile, data)
        stuelab.metagen.fromarrays(
            dfile,
            xarray=data['Frequency (Hz)'],
            yarray=[0],
            xtitle='Frequency (Hz)',
            ytitle='Index',
            colnames=list(data),
        )
finally:
    print("Shutting down devices...")
    # Enclose every closing attempt in its own try-except block to ensure that
    # we attempt to close them all even if there is a problem.
    try:
        vna.write('TRIG:SOUR IMM')
        vna.close()
    except: pass
