from mpi4py import *
import teca 
import numpy as np
import time
import sys
sys.path.append("PCA_ARs")
from PC_ivt import *
import cftime

class teca_sample_track(teca.teca_python_algorithm):
    """ This class illustrates how to sample mesh based data along a cylcone
    trajectory. The example will make a plot of the sampled data. The data is
    returned in numpy arrays and could be processed as needed instead of making
    a plot. The class is parallel and on NERSC it must be run on compute nodes.
    The class parallelizes over tracks. Each rank is given a unique set of
    tracks to process. Thus this example shows how to work with complete
    tracks.

    In the report_callback we inform the down stream about the number of tracks
    available. The sets things up for the map-reduce over tracks.

    In the request_callback we look up the requested track, get the lat lon
    coordinates of the track and generate a request to the NetCDF CF2 reader
    for a widow centered on each point of the track.

    In the execute_callback we are served the data and make a plot of each
    array.
    """
    def __init__(self):
        self.arrays = []
        self.output_prefix = './'
        self.track_reader = None
        self.track_table = None
        self.verbose = 0

    def set_input_connection(self, port, obj):
        if port == 0:
            self.track_reader = obj[0]
        else:
            self.impl.set_input_connection(obj)

    def set_arrays(self, val):
        """ Set a list of arrays to load """
        self.arrays = val

    def set_algorithm(self, val):
        """ algorithm identifier
            0: PCs
            1: Sample along PCs
        """
        self.algorithm = val
        
    def set_delta_lon(self, val):
        """ horizontal window size (15 deg) """
        self.delta_lon = float(val)

    def set_delta_lat(self, val):
        """ vertical window size (15 deg) """
        self.delta_lat = float(val)

    def set_output_prefix(self, val):
        """ Path prepended to output files """
        self.output_prefix = val

    def report(self, port, md_in):

        # verify that user supplied a pipeline for us to invoke
        # to get the track table
        if self.track_reader is None:
            sys.stderr.write('Error: track reader is None')
            return teca_meta_data()

        # skip executing the pipeline if we already have the track
        # table
        if self.track_table is None:

            # the capture will let us get the table
            dc = teca.teca_dataset_capture.New()
            dc.set_input_connection(self.track_reader.get_output_port())
            dc.update()

            # tables are usually only read on rank 0. broadcast columns
            # we need to the other ranks
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            if rank == 0:
                # get the track table. The 'as' functions convert from an
                # abstract dataset to a concrete one
                self.track_table = teca.as_teca_table(dc.get_dataset())
                # get the collumns
                ids = self.track_table.get_column('track_id').as_array()
                lon = self.track_table.get_column('lon').as_array()
                lat = self.track_table.get_column('lat').as_array()
                time = self.track_table.get_column('time').as_array()
                ar_id = self.track_table.get_column('ar_id').as_array()
                life = self.track_table.get_column('life').as_array()
                stage = self.track_table.get_column('stage').as_array()

                # send them to the other rnaks
                comm.bcast(ids, root=0)
                comm.bcast(lon, root=0)
                comm.bcast(lat, root=0)
                comm.bcast(time, root=0)
                comm.bcast(ar_id, root=0)
                comm.bcast(life, root=0)
                comm.bcast(stage, root=0)
            else:
                # receive the columns
                ids = comm.bcast(None, root=0)
                lon = comm.bcast(None, root=0)
                lat = comm.bcast(None, root=0)
                time = comm.bcast(None, root=0)
                ar_id = comm.bcast(None, root=0)
                life = comm.bcast(None, root=0)
                stage = comm.bcast(None, root=0)
               # put them in the table
                self.track_table = teca.teca_table.New()
                self.track_table.append_column('track_id', teca.teca_variant_array.New(ids))
                self.track_table.append_column('lon', teca.teca_variant_array.New(lon))
                self.track_table.append_column('lat', teca.teca_variant_array.New(lat))
                self.track_table.append_column('time', teca.teca_variant_array.New(time))
                self.track_table.append_column('ar_id', teca.teca_variant_array.New(ar_id))
                self.track_table.append_column('life', teca.teca_variant_array.New(life))
                self.track_table.append_column('stage', teca.teca_variant_array.New(stage))

        # build the report. the input metadata has a list of arrays
        # coordinates etc from the NetCDF files. copy these and replace
        # number of time steps with number of track ids since we are
        # parallelizing over track ids
        ids = self.track_table.get_column('track_id').as_array()
        uids = np.unique(ids)
        md_out = teca.teca_metadata(md_in[0])
        md_out['index_initializer_key'] = 'number_of_tracks'
        md_out['index_request_key'] = 'track_number'
        md_out['number_of_tracks'] = len(uids)

        return md_out

    def request(self, port, md_in, req_in):

        # this identifies which track is requested
        map_id = req_in['track_number']

        # we are going to request a window of data centered
        # on the track of dimension delta_lon x delta_lat deg
        # first step is get the track coordinates
        lon = self.track_table.get_column('lon').as_array()
        lat = self.track_table.get_column('lat').as_array()
        time = self.track_table.get_column('time').as_array()
        track_ids = self.track_table.get_column('track_id').as_array()
        life = self.track_table.get_column('life').as_array()
        ar_id = self.track_table.get_column('ar_id').as_array()
        stage = self.track_table.get_column('stage').as_array()
        
        track_uids = np.unique(track_ids)
        track_id = track_uids[map_id]

        # select just the track of interest
        ii = np.where(track_ids == track_id)[0]

        # select its coordinates
        track_lon = lon[ii]
        track_lat = lat[ii]
        track_time = time[ii]
        n_pts = len(ii)

        # now get the bounds of the data on disk
        mesh_coords = md_in[0]['coordinates']
        mesh_lon = mesh_coords['x']
        mesh_lat = mesh_coords['y']
        min_mesh_lon = np.min(mesh_lon)
        max_mesh_lon = np.max(mesh_lon)
        min_mesh_lat = np.min(mesh_lat)
        max_mesh_lat = np.max(mesh_lat)

        # 1/2 window size because we are centered on track
        d_lon = self.delta_lon#/2.0
        d_lat = self.delta_lat#/2.0
        d_lon = 60
        d_lat = 60

        # for each point on the track construct a request for a patch
        # centered on the track
        reqs_out = []
        for i in range(n_pts):

            lon = track_lon[i]
            lat = track_lat[i]
            time = track_time[i]

            # compute the window centered on the track
            # make sure that it doesn'time extend out side of the
            # available data
            window = [max(lon - d_lon, min_mesh_lon), \
                min(lon + d_lon, max_mesh_lon), max(lat - d_lat, min_mesh_lat), \
                min(lat + d_lat, max_mesh_lat), 0.0, 0.0]

            # create the new request by copying the incoming request
            req = teca.teca_metadata(req_in)

            # request the subset at this time
            req['bounds'] = window
            req['time'] = time
            req['arrays'] = self.arrays

            # add it to the requests
            reqs_out.append(req)

        return reqs_out

    def execute(self, port, data_in, req_in):
        """ Define the TECA execute phase for this algorithm.

            Outputs a TECA table row, which is intended to be used in conjunction
            with teca_table_reduce, and teca_table_sort.
                    # data we requested is served to us here.
                    # we will get scalar fiels as numpy arrays
                    # this is where you would do something useful
                    # with the data. here we will just plot it
        """
        # map index is the track id
        map_id = req_in['track_number']
           
        # get the mesh structure containing the input data for thecurrent timestep
        in_mesh = teca.as_teca_cartesian_mesh(data_in[0])
        if in_mesh is None:
            return teca.teca_table.New()

        # get the mesh metadata
        in_md = in_mesh.get_metadata()

        # get the unique track ids and look up the specific track id
        # that we will process using the map index
        ids = self.track_table.get_column('track_id').as_array()
        ar_ids = self.track_table.get_column('ar_id').as_array()
        lifes = self.track_table.get_column('life').as_array()
        stages = self.track_table.get_column('stage').as_array()
        uids = np.unique(ids)
        
        track_id = uids[map_id]
        ar_id = ar_ids[map_id]
        stage = stages[map_id]
        life = lifes[map_id]

        # get the storm position and size
        track_lon = self.track_table.get_column('lon').as_array()
        track_lat = self.track_table.get_column('lat').as_array()

        # get the lat/lon coordinates
        lat = in_mesh.get_y_coordinates()
        lon = np.array(in_mesh.get_x_coordinates())
        # shift longitudes by +360 if any negative longitudes are detected
        if lon.min() < 0:
            lon += 360 
        lon2d, lat2d = np.meshgrid(lon, lat)

        # Extract ivt array and its size
        nx,ny = len(lon),len(lat)
        array = np.reshape(in_mesh.get_point_arrays().get("ivt").as_array(), [ny,nx])
#         ivtx = np.reshape(in_mesh.get_point_arrays().get("ivtx").as_array(), [ny,nx])
#         ivty = np.reshape(in_mesh.get_point_arrays().get("ivty").as_array(), [ny,nx])
        arci = np.reshape(in_mesh.get_point_arrays().get("arci").as_array(), [ny,nx])

        # select the data for this track
        ii = np.where(ids == track_id)[0]
        track_lon = track_lon[ii]
        track_lat = track_lat[ii]

        # Verbose rank and track_id processing
        if self.verbose > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            sys.stderr.write('[rank %d processing track %d]'%(rank, track_id))

        # create the output table for statistica methods
        out_table = teca.teca_table.New()
        # set the time calendar/units based on the input times
        out_table.set_calendar(in_mesh.get_calendar())
        out_table.set_time_units(in_mesh.get_time_units())

        # initialize netCDF metadata
        out_atts = teca.teca_metadata()

        #=========================================   
        # set lat metadata
        mask_atts = teca.teca_array_attributes()
        mask_atts.long_name = "lat of AR centroid"
        mask_atts.units = "deg"
        out_atts["lat"] = mask_atts.to_metadata()

        # set lon metadata
        mask_atts = teca.teca_array_attributes()
        mask_atts.long_name = "lon of AR centroid"
        mask_atts.units = "deg"
        out_atts["lon"] = mask_atts.to_metadata()

        # set length metadata
        time_atts = teca.teca_array_attributes()
        time_atts.long_name = "AR_PC_length"
        time_atts.units = "km"
        time_atts.description = "AR length from principal component analysis"
        out_atts["length"] = time_atts.to_metadata()      

        # set width metadata
        time_atts = teca.teca_array_attributes()
        time_atts.long_name = "AR_PC_width"
        time_atts.units = "km"
        time_atts.description = "AR width from principal component analysis"
        out_atts["width"] = time_atts.to_metadata()      

        # set area metadata
        time_atts = teca.teca_array_attributes()
        time_atts.long_name = "AR_PC_area"
        time_atts.units = "km"
        time_atts.description = "AR area from principal component analysis"
        out_atts["area"] = time_atts.to_metadata()      
        
        # set track_id metadata
        time_atts = teca.teca_array_attributes()
        time_atts.long_name = "track_id for AR"
        time_atts.units = "dimensionless"
        time_atts.description = "track_id for each AR at each time step"
        out_atts["track_id"] = time_atts.to_metadata()

        # set ar_id metadata
        time_atts = teca.teca_array_attributes()
        time_atts.long_name = "Atmospheric River Identifier"
        time_atts.units = "dimensionless"
        time_atts.description = "ar_id unique identifier for each AR during its lifecycle"
        out_atts["ar_id"] = time_atts.to_metadata()        

        # set time metadata
        time_atts = teca.teca_array_attributes(in_md["attributes"]['time'])
        time_atts.long_name = "time"
        time_atts.calendar = in_mesh.get_calendar()
        out_atts["time"] = time_atts.to_metadata()

        # set d0 metadata
        mask_atts = teca.teca_array_attributes()
        mask_atts.long_name = "distance0 to AR centroid"
        mask_atts.description = "distance0 to AR centroid across AR (in the first PC direction)"
        mask_atts.units = "km"
        out_atts["d0"] = mask_atts.to_metadata()

        # set ivt0 metadata
        mask_atts = teca.teca_array_attributes()
        mask_atts.long_name = "ivt0 along d0"
        mask_atts.description = "IVT across AR (in the first PC direction)"
        mask_atts.units = "kg m-1 s-1"
        out_atts["ivt0"] = mask_atts.to_metadata()

        # set d1 metadata
        mask_atts = teca.teca_array_attributes()
        mask_atts.long_name = "distance1 to AR centroid"
        mask_atts.description = "distance1 to AR centroid across AR (in the second PC direction)"
        mask_atts.units = "km"
        out_atts["d1"] = mask_atts.to_metadata()

        # set ivt0 metadata
        mask_atts = teca.teca_array_attributes()
        mask_atts.long_name = "ivt1 along d1"
        mask_atts.description = "IVT across AR (in the second PC direction)"
        mask_atts.units = "kg m-1 s-1"
        out_atts["ivt1"] = mask_atts.to_metadata()

        # set life metadata
        time_atts = teca.teca_array_attributes()
        time_atts.long_name = "Life span of AR"
        time_atts.units = "time steps"
        time_atts.description = "Life span of each AR in model time steps. Can be converted to days using calendar and time units"
        out_atts["life"] = time_atts.to_metadata()      

        # set stage metadata
        time_atts = teca.teca_array_attributes()
        time_atts.long_name = "AR lifecycle stage"
        time_atts.units = "time steps"
        time_atts.description = "Stage over the life span of each AR (from 0 to life-1)"
        out_atts["stage"] = time_atts.to_metadata()      

#         #-----------------------------------------------------------------------------------------------------
#         # Apply PCs algorithm
#         y0, x0, theta, PC0, PC1, length, width, area = \
#                 create_weight(array, lat, lon, track_lat[0], track_lon[0], arci)
        
#         if self.algorithm == 0:
        # define the columns for algorithm 0
#         out_table.declare_columns(["track_id", "ar_id", "time", "lat", "lon", "life", "stage", "length", "width", "area"], \
#                                   ['i',         'i',     "d",    "d",   "d",    "i",    "i",       "d",     "d",    "d"] )  

#         y0, x0, theta, PC0, PC1, length, width, area = \
#                 create_weight(array, lat, lon, track_lat[0], track_lon[0], arci)

#         # add the time and ELI values to the output table
#         if length is not None:
#             out_table << track_id << ar_id << in_mesh.get_time() << y0 << x0 << life << stage << length << width << area
            
#         elif self.algorithm == 1:
#             # define the columns for algorithm 1
#             out_table.declare_columns(["track_id", "ar_id", "time", "lat", "lon", "life", "stage", "length", "width", "area", "d0", "ivt0",  "d1", "ivt1"], \
#                                       ['i',         'i',     "d",    "d",   "d",    "i",    "i",       "d",     "d",    "d",  "d",   "d"  ,   "d",   "d"] )  
# #             out_table.declare_columns(["track_id", "ar_id", "time", "d0", "ivt0",  "d1", "ivt1", "life", "stage"], \
# #                                       ['i',         'i',     "d",    "d",   "d"  ,   "d",   "d",   "i",    "i"] )  
        
#             #-----------------------------------------------------------------------------------------------------
#             # Sample IVT along PCs
#             if length is not None:
#                 dist0,ivt0,dist1,ivt1 = Sample_ivt_pc(array, lat, lon, y0, x0, PC0, PC1)
        
#             if dist0 is not None:
#                 for d0, i0, d1, i1 in zip(dist0, ivt0, dist1, ivt1):
#                     out_table << track_id << ar_id << in_mesh.get_time() << y0 << x0 << life << stage << length << width << area << d0 << i0 << d1 << i1
# #                     out_table << track_id << in_mesh.get_time() << ar_id << d0 << i0 << d1 << i1 << life << stage

#         #-----------------------------------------------------------------------------------------------------
        out_table.declare_columns(["track_id", "ar_id", "time", "lat", "lon", "life", "stage", "length", "width", "area", "d0", "ivt0",  "d1", "ivt1"], \
                                  ['i',         'i',     "d",    "d",   "d",    "i",    "i",       "d",     "d",    "d",  "d",   "d"  ,   "d",   "d"] )  
 
        # Apply PCs algorithm
        y0, x0, theta, PC0, PC1, length, width, area = \
                create_weight(array, lat, lon, track_lat[0], track_lon[0], arci)
        
        # Sample IVT along PCs
        if length is not None:
            dist0,ivt0,dist1,ivt1 = Sample_ivt_pc(array, lat, lon, y0, x0, PC0, PC1)

            if dist0 is not None:
                c = 0
                for d0, i0, d1, i1 in zip(dist0, ivt0, dist1, ivt1):
                    if c==0:
                        out_table << track_id << ar_id << in_mesh.get_time() << y0 << x0 << life << stage << length << width << area << d0 << i0 << d1 << i1
                    else:
                        out_table << track_id << ar_id << in_mesh.get_time() << y0 << x0 << life << stage << np.nan << np.nan << np.nan << d0 << i0 << d1 << i1
                    c += 1

        # RETURN CALCULATED VALUES AS A Teca TABLE
        return out_table

#=======================================================================================================================
#=======================================================================================================================
# MAIN FUNCTION

if __name__ == "__main__":
    t0 = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # process command line options
    argc = len(sys.argv)
    if argc < 5:
        if rank == 0:
           sys.stderr.write('Usage:\n' \
#              'python sample_tracks.py [delta lon] [delta lat] [track file]\n' \
#              '[mesh files regex] [output prefix] [surface pressure array] \n\n'\
                           )
        sys.exit()

    delta_lon = sys.argv[1]
    delta_lat = sys.argv[2]
    track_file = sys.argv[3]
    files_regex = sys.argv[4]
    output_prefix = sys.argv[5]
    run = sys.argv[6]
#     algorithm = int(sys.argv[7])

    # start with the NetCDF CF-2 reader. this will serve up
    # mesh based data subset on a window centered on the track
    cfr = teca.teca_multi_cf_reader.New()
    cfr.set_input_file(files_regex)
    
    # add stage to compute surface wind speed
#     ivtx = teca.teca_evaluate_expression.New()
#     ivtx.set_input_connection(cfr.get_output_port())
#     ivtx.set_result_variable('ivtx')
#     ivtx.set_expression('%s'%("uhusavi"))

#     ivty = teca.teca_evaluate_expression.New()
#     ivty.set_input_connection(ivtx.get_output_port())
#     ivty.set_result_variable('ivty')
#     ivty.set_expression('%s'%("vhusavi"))

    ivtt = teca.teca_evaluate_expression.New()
    ivtt.set_input_connection(cfr.get_output_port())
    ivtt.set_result_variable('ivt')
    ivtt.set_expression('(%s*%s+%s*%s)**0.5'%("uhusavi","uhusavi","vhusavi","vhusavi"))  

    arcid = teca.teca_evaluate_expression.New()
    arcid.set_input_connection(ivtt.get_output_port())
    arcid.set_result_variable('arci')
    arcid.set_expression('%s'%("ar_confidence_index"))  
        
#     # add stage to compute vorticity
#     vort = teca_vorticity.New()
#     vort.set_input_connection(ctemp.get_output_port())
#     vort.set_component_0_variable(mb850_wind_var_0)
#     vort.set_component_1_variable(mb850_wind_var_0)
#     vort.set_vorticity_variable('vorticity_850mb')

#     # the table reader will load a set of tracks to process
    tr = teca.teca_table_reader.New()
    tr.set_file_name(track_file)

    expr=""
    tip = tr
    if expr:
        rr = teca.teca_table_remove_rows.New()
        rr.set_input_connection(tr.get_output_port())
        rr.set_mask_expression(expr)
        tip = rr

#     # Add the sample track filter. This loads a set of tracks
#     # and requests subsets of mesh based data on windows centered
#     # on the track.
#     arrays = ['ivt_zonal','ivt_meridional','ivt_squared','ivt']
    arrays = ['ivt','arci']
#      arrays += other_var

    st = teca_sample_track.New()
    st.set_input_connection(0, tip.get_output_port())
    st.set_input_connection(1, arcid.get_output_port())
    st.set_delta_lon(delta_lon)
    st.set_delta_lat(delta_lat)
    st.set_output_prefix(output_prefix)
    st.set_arrays(arrays)
#     st.set_algorithm(algorithm)
   
    # Add the map-reduce stage. this parallelizes the run over
    # tracks. Each MPI rank will be given a unique set of tracks
    # to process
    mr = teca.teca_table_reduce.New()
    mr.set_input_connection(st.get_output_port())
    mr.set_thread_pool_size(1)
  
    # sort
    tsort = teca.teca_table_sort.New()
    tsort.set_input_connection(mr.get_output_port())
    tsort.set_index_column("track_id")
#     pipeline_stages.append(tsort)

    # Define output file name
#     output_filename="{}/{}_alg{}.nc".format(output_prefix,run,algorithm)
    output_filename="{}/{}.nc".format(output_prefix,run)
    
    # writer
    tfw = teca.teca_table_writer.New()
    tfw.set_input_connection(tsort.get_output_port())
    tfw.set_file_name(output_filename)
    tfw.set_row_dim_name("track_id")
#     pipeline_stages.append(tfw)

    # run the pipeline
    tfw.update()

    # report run time
    t1 = time.time()
    if rank == 0:
       sys.stderr.write('\ntotal run time: %0.2f seconds\n'%(t1 - t0))
    
    
    
    