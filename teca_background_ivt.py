#!/usr/bin/env python3
"""Calculates the IVT background valueude on CMIP-like data.
"""

from mpi4py import MPI
import teca
from calculate_background_ivt import calculate_background_ivt
import numpy as np
import time
from netCDF4 import num2date,date2num

if MPI.COMM_WORLD.Get_rank() == 0:
    print(f"Running with {MPI.COMM_WORLD.Get_size()} ranks.")

class teca_background_ivt(teca.teca_python_algorithm):
    """A TECA algorithm for calculating the ENSO Longitude Index."""

    output_variable_name = "IVT_bk"
    input_variable_name = "ivt"
    mask_variable_name = "ar_confidence_index"

    def set_arrays(self, val):
        """ Set a list of arrays to load """
        self.arrays = val
        
    def set_background(self, background):
        """ Set the name of the ivt variable. """
        self.background = background
        
    def set_input_variable_name(self, input_variable_name):
        """ Set the name of the ivt variable. """
        self.input_variable_name = input_variable_name

    def set_output_variable_name(self, output_variable_name):
        """ Set the name of the output (IVT background) variable. """
        self.output_variable_name = output_variable_name

    def set_mask_variable_name(self, mask_variable_name):
        """ Set the name of the variable containing the ivt/AR mask. """
        self.mask_varible_name = mask_variable_name

    def request(self, port, md_in, req_in):
        """ Define the TECA request phase for this algorithm"""
        req = teca.teca_metadata(req_in)
        req['arrays'] = self.input_variable_name

        # set the extent, which is required by 
        # teca_cartesian_mesh_regrid
        req['extent'] = teca.teca_metadata(md_in[0])['whole_extent']

        return [req]

    def report(self, port, md_in):
        """ Define the TECA report phase for this algorithm"""
        report_md = teca.teca_metadata(md_in[0])

        return report_md

    def execute(self, port, data_in, req):
        """ Define the TECA execute phase for this algorithm.

            Outputs a TECA table row, which is intended to be used in conjunction
            with teca_table_reduce, and teca_table_sort.
        
        """
        
        # get the mesh structure containing the input data for the
        # current timestep
        in_mesh = teca.as_teca_cartesian_mesh(data_in[0])

        # get the mesh metadata
        in_md = in_mesh.get_metadata()

        # create the output table
        out_table = teca.teca_table.New()
        # set the time calendar/units based on the input times
        out_table.set_calendar(in_mesh.get_calendar())
        out_table.set_time_units(in_mesh.get_time_units())
        
        # initialize netCDF metadata
        out_atts = teca.teca_metadata()

        # set time metadata
        time_atts = teca.teca_array_attributes(in_md["attributes"]['time'])
        time_atts.long_name = "time"
        time_atts.calendar = in_mesh.get_calendar()
        out_atts["time"] = time_atts.to_metadata()
        
        # set IVT_bk1 metadata
        if self.background == 1:
            ivt_atts = teca.teca_array_attributes()
            ivt_atts.long_name = "Background IVT (v1)"
            ivt_atts.units = "kg m-1 s-1"
            ivt_atts.description = "Sampled IVT from everywhere without filtering"
            out_atts["IVT_bk"] = ivt_atts.to_metadata()

        # set IVT_bk2 metadata
        if self.background == 2:
            ivt_atts = teca.teca_array_attributes()
            ivt_atts.long_name = "Background IVT (v2)"
            ivt_atts.units = "kg m-1 s-1"
            ivt_atts.description = "Sample IVT without filtering, except where there ar ARs"
            out_atts["IVT_bk"] = ivt_atts.to_metadata()

        # set IVT_bk3 metadata
        if self.background == 3:
            ivt_atts = teca.teca_array_attributes()
            ivt_atts.long_name = "Background IVT (v3)"
            ivt_atts.units = "kg m-1 s-1"
            ivt_atts.description = "Sampled IVT from everywhere after filtering the equator using gaussian band"
            out_atts["IVT_bk"] = ivt_atts.to_metadata()

        # set IVT_bk4 metadata
        if self.background == 4:
            ivt_atts = teca.teca_array_attributes()
            ivt_atts.long_name = "Background IVT (v4)"
            ivt_atts.units = "kg m-1 s-1"
            ivt_atts.description = "Sampled IVT from places without AR after filtering the equator using gaussian band"
            out_atts["IVT_bk"] = ivt_atts.to_metadata()
        
        ivt_atts = teca.teca_array_attributes()
        ivt_atts.long_name = "year"
        ivt_atts.units = ""
        ivt_atts.description = "year corresponding to the sampled ivt"
        out_atts["year"] = ivt_atts.to_metadata()
            
        # set the metadata
        out_table.get_metadata()["attributes"] = out_atts

        # get the lat/lon coordinates
        lat = in_mesh.get_y_coordinates()
        lon = np.array(in_mesh.get_x_coordinates())
        # shift longitudes by +360 if any negative longitudes are detected
        if lon.min() < 0:
            lon += 360 
        lon2d, lat2d = np.meshgrid(lon, lat)

        # Extract ivt array and its size
        nx,ny = len(lon),len(lat)
        ivt = np.reshape(in_mesh.get_point_arrays().get("ivt").as_array(), [ny,nx])#[np.newaxis, ...]
        arci = np.reshape(in_mesh.get_point_arrays().get("arci").as_array(), [ny,nx])#[np.newaxis, ...]

        # define the columns
#         out_table.declare_columns(["time", "IVT_bk", "year"], ['d', 'd', 'i'])
        out_table.declare_columns(["time", "IVT_bk"], ['d', 'd'])
#         print(num2date(in_mesh.get_time(), units=in_mesh.get_time_units(), calendar=in_mesh.get_calendar()), flush=True)
#         year = int(str(in_mesh.get_time()).split("-")[0])
#         date = num2date(in_mesh.get_time(), units=in_mesh.get_time_units(), calendar=in_mesh.get_calendar())
#         year = int(str(date).split("-")[0])
#         time = date2num(date, units=in_mesh.get_time_units(), calendar=in_mesh.get_calendar())
    
        ivt_bk = calculate_background_ivt(ivt, lat2d, lon2d, arci, self.background)

        # add the time and ivt_background value
        for i in range(len(ivt_bk)):
#             out_table << time << ivt_bk[i] << year
            out_table << in_mesh.get_time() << ivt_bk[i] 

#         # calculate background ivt
#         ivt_bk1,ivt_bk2,ivt_bk3,ivt_bk4 = calculate_background_ivt(ivt, lat2d, lon2d, arci)
    
#         # define the columns
#         out_table.declare_columns(["time", "IVT_bk1", "IVT_bk2", "IVT_bk3", "IVT_bk4"], ['d', 'd', 'd', 'd', 'd'])

#         # calculate background ivt
#         ivt_bk1,ivt_bk2,ivt_bk3,ivt_bk4 = calculate_background_ivt(ivt, lat2d, lon2d, arci)
        
#         # add the time and ivt_background value
#         for i in range(len(ivt_bk1)):
#             out_table << in_mesh.get_time() << ivt_bk1[i] << ivt_bk2[i] << ivt_bk3[i] << ivt_bk4[i]

#         # define the columns
#         out_table.declare_columns(["time", "IVT_bk"], ['d', 'd'])
        
#         # calculate background ivt
#         ivt_bk3 = calculate_background_ivt(ivt, lat2d, lon2d, arci)
        
#         # add the time and ivt_background value
#         for i in range(len(ivt_bk3)):
#             out_table << in_mesh.get_time() << ivt_bk3[i]

        # return the current table row
        return out_table

def construct_teca_pipeline(\
        files_regex,
        output_filename,
        nthreads = 1,
        be_verbose = False,
        background = 1,
        ):
    """Construct the TECA pipeline for this application."""

    # initialize the pipeline stages
    pipeline_stages = []

    # ivt reader
    cfr = teca.teca_multi_cf_reader.New()
    cfr.set_input_file(files_regex)
    pipeline_stages.append(cfr)

    ivtt = teca.teca_evaluate_expression.New()
#     ivtt.set_input_connection(cfr.get_output_port())
    ivtt.set_result_variable('ivt')
    ivtt.set_expression('(%s*%s+%s*%s)**0.5'%("uhusavi","uhusavi","vhusavi","vhusavi"))  
    pipeline_stages.append(ivtt)
    
    # AR confidence index reader
    arcid = teca.teca_evaluate_expression.New()
#     arcid.set_input_connection(ivtt.get_output_port())
    arcid.set_result_variable('arci')
    arcid.set_expression('%s'%("ar_confidence_index"))  
    pipeline_stages.append(arcid)
    
    st = teca_background_ivt.New()
#     st.set_input_connection(arcid.get_output_port())
    arrays = ['ivt','arci']
    st.set_arrays(arrays)
    st.set_background(background)
    pipeline_stages.append(st)
    
################################################################################
#CODE FROM TECA ELI TRAVIS' PROGRAM
#----------------------------------
#     # combine the mask file and variable readers
#     # ... the regrid function does nothing if the meshes
#     # are already on the same grid
#     combiner = teca.teca_cartesian_mesh_regrid.New()
#     combiner.set_input_connection(0, arcid.get_output_port())
#     pipeline_stages.append(combiner)

#     # Normalize coordinates
#     norm = teca.teca_normalize_coordinates.New()
#     # make sure longitude starts a 0
#     norm.set_enable_periodic_shift_x(1)
#     pipeline_stages.append(norm)

# #     # IVT background calculation
# #     ivt_bk = teca_evaluate_expression.New()
# #     ivt_bk.set_input_variable_name(input_variable_name)
# #     ivt_bk.set_output_variable_name(output_variable_name)
# #     pipeline_stages.append(ivt_bk)

#     reduction
#     map_reduce = teca.teca_table_reduce.New()
# #     map_reduce.set_thread_pool_size(nthreads)
# #     map_reduce.set_verbose(int(be_verbose))
# #     map_reduce.set_start_index(0)
# #     map_reduce.set_end_index(640)
#----------------------------------
#CODE FROM TECA ELI TRAVIS' PROGRAM
################################################################################

    map_reduce = teca.teca_table_reduce.New()
#     map_reduce.set_input_connection(st.get_output_port())
    map_reduce.set_verbose(0)
#     map_reduce.set_start_index(0)
#     map_reduce.set_end_index(81758)
    map_reduce.set_thread_pool_size(1)
    pipeline_stages.append(map_reduce)

    # sort
    tsort = teca.teca_table_sort.New()
#     tsort.set_input_connection(map_reduce.get_output_port())
    tsort.set_index_column("time")
    pipeline_stages.append(tsort)

    # writer
    tfw = teca.teca_table_writer.New()
#     tfw.set_input_connection(tsort.get_output_port())
    tfw.set_file_name(output_filename)
    tfw.set_row_dim_name("time")
    pipeline_stages.append(tfw)
    

#     connect the pipeline
    for n in range(1,len(pipeline_stages)):
        pipeline_stages[n].set_input_connection(\
                pipeline_stages[n-1].get_output_port())

    # return the last stage of the pipeline
    return pipeline_stages[-1]


if __name__ == "__main__":
    
    t0 = time.time()
    
    import argparse
    
    # construct the command line arguments
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        '--input_regex',
        help = "A regex file that points to the files containing ivtx, ivty, and arci (uhusavi, vhusavi, ar_confidence_index)",
        )
    parser.add_argument(
        '--output_file',
        help = "The name of the file to write to disk." ,
        )
    parser.add_argument(
            '--nthreads',
        help = "The number of threads to use (-1 = TECA determines the thread count)",
        default = 1,
        )
    parser.add_argument(
        '--verbose',
        help = "Indicates whether to turn on verbose output.",
        default = False,
        )
    parser.add_argument(
        '--background',
        help = "Indicates which background to calculate (1-4)",
        default = 4,
        )
    
    # parse the command line arguments
    args = parser.parse_args()

    # construct the TECA pipeline
    pipeline = construct_teca_pipeline(
        files_regex = args.input_regex,
        output_filename = args.output_file,
        nthreads = int(args.nthreads),
        be_verbose = bool(args.verbose),
        background = int(args.background),
    )

    teca.teca_profiler_initialize()
    # run the pipeline
    pipeline.update()
    teca.teca_profiler_finalize()


    # report run time
    t1 = time.time()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('\ntotal run time: %0.2f seconds\n'%(t1 - t0))
    
    
    
    