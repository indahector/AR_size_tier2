#!/python

from ARsize_cmip56_library import *

for run in runs:

    areas50,counts50,areas67,counts67,years = calculate_arci_areas(run)
    create_arci_area_netcdf(run,areas50,counts50,areas67,counts67,years)

