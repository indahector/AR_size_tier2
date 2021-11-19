using_mpi = True
try:
    if not using_mpi:
        import mpi4py
        mpi4py.rc(initialize=False, finalize=False)
        raise ImportError()
    else:
        from mpi4py import *
        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
except ImportError:
    rank = 0
    n_ranks = 1
    
import numpy as np
# import pandas as pd
import argparse
import numba
import teca

teca.set_stack_trace_on_error()
teca.set_stack_trace_on_mpi_error()

@numba.njit()
def assign_component_var(component_var, component_ids, component_var_out):
    """ Assign component_var, taking care to remove the background """
    i = 0
    for n in range(len(component_var)):
        if component_ids[n] != 0:
            component_var_out[i] = component_var[n]
            i += 1


class SegmentationProps(teca.teca_python_algorithm):
    """ A custom TECA algorithm for getting properties of segmented fields

    """

    segmentation_var = "cc"
    max_area_count = 128

    def set_segmentation_var(self, var):
        self.segmentation_var = var

    def set_max_area_count(self, count):
        self.max_area_count = count

    def report(self, port, input_md):

        md = teca.teca_metadata(input_md[0])
        variables = md["variables"]
        variables.append("count")
        variables.append("areas")
        md["variables"] = variables

        count_atts = teca.teca_array_attributes( \
                teca.teca_long_array_code.get(),
                teca.teca_array_attributes.no_centering,
                1, "count", "number of segmented objects", "", 0)


        area_atts = teca.teca_array_attributes( \
                teca.teca_double_array_code.get(),
                teca.teca_array_attributes.no_centering,
                self.max_area_count, "m**2", "area of segmented objects", "", 0)

        area_atts_md = area_atts.to_metadata()
        area_atts_md['_FillValue'] = 0.0
        atts = md["attributes"]
        atts["count"] = count_atts.to_metadata()
        atts["areas"] = area_atts_md
        md["attributes"] = atts

        return md
    
    def request(self, port, input_md, request):

        # copy the request metadata
        req = teca.teca_metadata(request)

        # pass the incoming request upstream; add what we need
        arrays = req["arrays"]
        if type(arrays) is str:
            arrays = [arrays]
        if self.segmentation_var not in arrays:
            arrays.append(self.segmentation_var)

        # remove what we produce
        try:
            arrays.remove("count")
            arrays.remove("areas")
        except:
            pass

        req["arrays"] = arrays

        return [req]


    def execute(self, port, input_data, request):
        md = input_data[0].get_metadata()

        # get the counts
        count = np.array(md["number_of_components"], dtype = int)
        count -= 1 # decrement the count to remove the background
        count = teca.teca_variant_array.New(count)

        area_out = np.zeros(self.max_area_count)
        areas = np.atleast_1d(np.array(md["component_area"]))
        component_ids = np.atleast_1d(np.array(md["component_ids"]))
        assign_component_var(areas, component_ids, area_out)
        area_out = teca.teca_variant_array.New(area_out)

        mesh = teca.as_teca_cartesian_mesh(input_data[0])
        info_arrays = mesh.get_information_arrays()
        info_arrays.append("count", count)
        info_arrays.append("areas", area_out)

        return mesh



def construct_teca_pipeline(run,
                            seg_array = "ar_confidence_index",
                            num_threads = 1,
                            steps_per_file = 1000000,
                            low_seg_threshold = 0.67,
                            high_seg_threshold = 2.0,
                            output_template = "teca_segmentation_prop%t%.nc"):
    """ Constructs a teca pipeline.

        input:
        ------

            run           : the run name
            
            seg_array     : the array to segment
            
            num_threads     : number of threads
            
            steps_per_file  : the number of steps to write per file

            low_seg_threshold : the lower-bound threshold to apply to the 
                                segmented field

            high_seg_threshold : the upper-bound threshold to apply to the 
                                 segmented field

            output_template : a template for output filenames (the string %t%
                              inserts a timestamp into the file name)

        output:
        -------

            pipeline : a teca pipeline



    """

    pipeline_stages = []
    
#     if rank == 0:
        
#         with open("metadata_tier2.csv", "r") as fin:
#             lines = fin.readlines()
#             lines = lines[1:]
        
#         for line in lines:
#             if run in line:
#                 rr,ntime,t0,units,calendar = line.split(",")
#                 ntime,t0,calendar = int(ntime),float(t0),calendar.split("\n")[0]
        
#         times = np.cumsum(0.25*np.ones(ntime)) - 0.25 + t0
#         timeinfo = (calendar, units, times)
#     else:
#         timeinfo = None

#     # get the time info on all processors
#     if using_mpi:
#         timeinfo = MPI.COMM_WORLD.bcast(timeinfo, root = 0)
#         calendar, units, times = timeinfo

#     if rank == 0:
#         print(times)

    # cf reader
    reader = teca.teca_cf_reader.New()
    reader.set_files_regex(f"/global/homes/i/indah/RESEARCH/ARTMIP/tier2/CMIP56/Means/{run}/{run}.ar_confidence_index.*")
    # reader.set_calendar(calendar)
    # reader.set_t_units(units)
    # reader.set_t_values(times.astype(np.float64))
    # reader.set_metadata_cache_dir("./")
    pipeline_stages.append(reader)
    
    # coordinate normalizer
    coords = teca.teca_normalize_coordinates.New()
    pipeline_stages.append(coords)

    # segmentation
    seg = teca.teca_binary_segmentation.New()
    seg.set_threshold_variable(seg_array)
    seg.set_segmentation_variable("seg")
    seg.set_low_threshold_value(low_seg_threshold)
    seg.set_high_threshold_value(high_seg_threshold)
    pipeline_stages.append(seg)

    # connected components
    cc = teca.teca_connected_components.New()
    cc.set_segmentation_variable("seg")
    cc.set_component_variable("cc")
    pipeline_stages.append(cc)

    area = teca.teca_2d_component_area.New()
    area.set_component_variable("cc")
    area.set_contiguous_component_ids(1)
    pipeline_stages.append(area)

    # segmentation props
    props = SegmentationProps.New()
    pipeline_stages.append(props)

    # index (timestep) executive
    exe = teca.teca_index_executive.New()

    # cf writer
    writer = teca.teca_cf_writer.New()
    writer.set_thread_pool_size(num_threads)
    #writer.set_point_arrays(["cc"])
    writer.set_information_arrays(["count", "areas"])
    # writer.set_information_arrays(["areas"])
    writer.set_file_name(output_template)
    writer.set_steps_per_file(steps_per_file)
    writer.set_executive(exe)
    writer.set_verbose(1)
    # writer.set_layout("number_of_steps")
    writer.set_layout_to_number_of_steps()
    pipeline_stages.append(writer)

    # stitch the pipeline together
    for n in range(1,len(pipeline_stages)):
        pipeline_stages[n].set_input_connection(
                                         pipeline_stages[n-1].get_output_port())


    return pipeline_stages[-1]

   
    # parse the command line
parser = argparse.ArgumentParser(
    description='Reduce the time axis of a NetcCDF CF2 dataset '
                'using a predfined interval and reduction operator')

parser.add_argument('run', type=str, nargs = 1,
                    help='ARTMIP run string')

parser.add_argument('--steps_per_file', type=int, default=1000000,
                    help='number of time steps to write to each output '
                         'file. (12)')

parser.add_argument('--n_threads', type=int, default=1,
                    help='Number of threads to use when stremaing the '
                         'reduction (1)')

parser.add_argument('--output_file', type=str, required=True,
                    help='file pattern for writing output netcdf '
                         'files. %%t%% will be replaced by a date/time '
                         'string or time step. See the teca_cf_writer '
                         'for more information.')

# prevent spew when running under mpi
try:
    args = parser.parse_args()
except Exception:
    if rank == 0: raise

out_file = args.output_file
steps_per_file = args.steps_per_file
n_threads = args.n_threads
run = args.run[0]

# construct the TECA pipeline
pipeline = construct_teca_pipeline(
                                   run,
                                   num_threads = n_threads,
                                   steps_per_file = steps_per_file,
                                   output_template = out_file,
                                  )
# TODO: Need to understand why we are only counting one AR in ARCONNECT v2

# run the pipeline
if pipeline is not None:
    pipeline.update()

 

