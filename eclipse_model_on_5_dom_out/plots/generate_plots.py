import sys
import glob
import xlrd
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.dates as mdates

from netCDF4 import Dataset
import wrf

def get_observed_series(data_sheet):
    time_col = 1
    temp_col = 6
    rad_col = 9
    wind_mag_col = 10
    wind_dir_col = 12

    eclipse_day = dt.datetime(2017, 8, 21)
    time = [eclipse_day + dt.timedelta(days=data_sheet.cell_value(i, time_col)) for i in range(1, data_sheet.nrows)]
    temp = [data_sheet.cell_value(i, temp_col) for i in range(1,data_sheet.nrows)]
    rad = [data_sheet.cell_value(i, rad_col) for i in range(1,data_sheet.nrows)]
    wspd = [data_sheet.cell_value(i, wind_mag_col) for i in range(1,data_sheet.nrows)]
    wdir = [data_sheet.cell_value(i, wind_dir_col) for i in range(1,data_sheet.nrows)]

    time = pd.to_datetime(time)
    return time, temp, rad, wspd, wdir

def to_local_time(var_data):
    start_time = var_data["Time"][0].dt.strftime("%c").item()
    end_time = var_data["Time"][-1].dt.strftime("%c").item()
    periods = len(var_data["Time"])
    local_time = pd.date_range(start=start_time, end=end_time, periods=periods, tz='UTC')
    local_time = local_time.tz_convert('US/Central')

    # Strip the time zone here because we need the underlying values to be changed to 
    # our local timezone, and converting timezone-aware datetimes to numpy.datetime64 is deprecated
    local_time = local_time.tz_localize(None) 
    var_data = var_data.assign_coords(local_time=("Time", local_time.values))
    return var_data.set_index(Time="local_time")

def save_plot(ax, title, y_label, x_label, var_name, plot_type_name, plot_time_range, filename_in, output_dir):
    # Points of interest to mark on plot (In CDT)
    sunrise = dt.datetime(2017, 8, 21, 6, 10, 32)
    eclipse_start = dt.datetime(2017, 8, 21, 11, 57)
    totality_start = dt.datetime(2017, 8, 21, 13, 26)
    totality_end = dt.datetime(2017, 8, 21, 13, 28)
    eclipse_end = dt.datetime(2017, 8, 21, 14, 52)
    sunset = dt.datetime(2017, 8, 21, 19, 30, 16)

    markers = [eclipse_start, totality_start, totality_end, eclipse_end]
    #markers = [sunrise, eclipse_start, totality_start, totality_end, eclipse_end, sunset]
    for mark in markers:
        ax.axvline(mark,color='r', linewidth=0.5)

    # Format
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend()
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.tight_layout()

    # Generate filename
    domain_level = filename_in.split('/')[-1].split('_')[1] 
    start_time = plot_time_range[0].strftime("%d%b%Y_%H%M")
    end_time = plot_time_range[-1].strftime("%d%b%Y_%H%M")
    fig_name = var_name + "_" + plot_type_name + "_" + domain_level + "_" + start_time + "to" +  end_time + ".png"
    print("\t" + fig_name)
    plt.savefig(output_dir + fig_name)

def main():
    # Get files from command line or take hard-coded folder.
    # Arguements are optional, but if one is specified, they all should be
    # vertical_profile_plots.py [input_pattern] [output_directory] [num_threads]
    # you can use a wildcard pattern:
    # i.e., python vertical_profile_plots.py ../output_files/wrfout* ../plots/ 8
    # or you can list the input files:
    # i.e., python vertical_profile_plots.py ../output_files/wrfout_d01 ../output_files/wrfout_d02 ../plots/ 1
    if len(sys.argv) > 1: 
        filenames = sys.argv[1:-2] #glob.glob(sys.argv[1])
        print(filenames)
        output_dir = sys.argv[-2]
        wrf.omp_set_num_threads(int(sys.argv[-1]))
    else:
        filenames = glob.glob("/project/ssmith_uksr/WRF_ARW/cases/eclipse_2017/eclipse_model_on_5_dom_out/wrfout_d01*")
        output_dir = ''

    # Get data from published sensor data
    ws_workbook = xlrd.open_workbook('/project/ssmith_uksr/WRF_ARW/DATA/2017_eclipse_observed/Weather_Station_data.xlsx')
    ws_first_sheet = ws_workbook.sheet_by_index(0)
    tower_workbook = xlrd.open_workbook('/project/ssmith_uksr/WRF_ARW/DATA/2017_eclipse_observed/Tower_data.xlsx')
    tower_first_sheet = tower_workbook.sheet_by_index(0)
    soil_workbook = xlrd.open_workbook('/project/ssmith_uksr/WRF_ARW/DATA/2017_eclipse_observed/Soil_data.xlsx')
    soil_first_sheet = soil_workbook.sheet_by_index(0)

    time_ws, temp_ws, rad_ws, wspd_ws, wdir_ws = get_observed_series(ws_first_sheet)
    time_tower, temp_tower, _ , wspd_tower, wdir_tower = get_observed_series(tower_first_sheet)
    time_soil, temp_soil, _ , _ , _ = get_observed_series(soil_first_sheet)

    # Coordinates to take sample from
    center_lat = 36.797326
    center_lon = -86.812341

    for filename in sorted(filenames):
        print(filename)
        #Structure the WRF output file
        ncfile = Dataset(filename)

        #Extract data from WRF output files
        tc = wrf.getvar(ncfile, "tc", wrf.ALL_TIMES) # Atmospheric temperature in celsius
        t2 = wrf.getvar(ncfile, "T2", wrf.ALL_TIMES) # Temperature at 2 m, in Kelvin
        # Convert T2 to degrees C
        t2 = t2 - 273.15
        t2.attrs["units"] = "degC"
        theta = wrf.getvar(ncfile, "theta", wrf.ALL_TIMES, units="degC")
        rh = wrf.getvar(ncfile, "rh", wrf.ALL_TIMES)
        wspd_wdir =  wrf.getvar(ncfile, "uvmet_wspd_wdir",  wrf.ALL_TIMES)
        # Split wind speed and direction
        wspd = wspd_wdir[0,:,:,:,:]
        wdir = wspd_wdir[1,:,:,:,:]
        

        # These variables aren't included in getvar, so have to be extracted manually
        swdown = wrf.extract_vars(ncfile, wrf.ALL_TIMES, "SWDOWN").get('SWDOWN')
        gnd_flx = wrf.extract_vars(ncfile, wrf.ALL_TIMES, "GRDFLX").get('GRDFLX')

        #Create Dictionary to associate quanitity names with the corresponding data
        two_dim_vars = {'swdown':swdown, 'gnd_flx':gnd_flx, 'T2':t2}
        three_dim_vars = {'tc':tc, 'theta': theta, 'rh':rh, 'wspd':wspd}
        

        #Get the grid coordinates from our earth lat/long coordinates
        center_x, center_y = wrf.ll_to_xy(ncfile, center_lat, center_lon)
        
        # Plot all 3D variables over time
        for var_name, var_data in three_dim_vars.items():
            # Convert to Local Time
            var_data = to_local_time(var_data)

            # Get data frequency
            freq = pd.Timedelta(var_data["Time"][1].values - var_data["Time"][0].values)
            
            # Interpolate to height above ground level
            try:
                var_data_agl = wrf.vinterp(ncfile, var_data, 'ght_agl', np.linspace(0, 0.1, 100), field_type=var_name ,timeidx=wrf.ALL_TIMES)
            except ValueError:
                var_data_agl = wrf.vinterp(ncfile, var_data, 'ght_agl', np.linspace(0, 0.1, 100), field_type='none' ,timeidx=wrf.ALL_TIMES)
            
            # Convert height to meters
            var_data_agl["interp_level"] = var_data_agl["interp_level"] * 1000

            # Time ranges
            plot_time_ranges = []
            plot_time_range1 = pd.date_range(start="2017-08-21T10:24:00", end="2017-08-21T14:33:00", freq=freq)
            plot_time_range1 = plot_time_range1.floor(freq)
            plot_time_range2 = pd.date_range(start="2017-08-21T13:09:00", end="2017-08-21T14:33:00", freq=freq)
            plot_time_range2 = plot_time_range2.floor(freq)
            plot_time_range3 = pd.date_range(start="2017-08-21T09:45:00", end="2017-08-21T15:00:00", freq=freq)
            plot_time_range3 = plot_time_range3.floor(freq)

            plot_time_ranges = [plot_time_range1, plot_time_range2, plot_time_range3]

            # Vertical Profile Plots
            for plot_time_range in [rng for rng in plot_time_ranges if len(rng) > 1]:
                fig, ax = plt.subplots()
                var_data_agl.isel(south_north=center_y, west_east=center_x).sel(Time=plot_time_range, method="nearest").plot(ax=ax, x="Time")
                save_plot(ax=ax, 
                    title='', 
                    y_label="z (m)", 
                    x_label="Local Time (CDT)", 
                    var_name=var_name, 
                    plot_type_name="vertical_profile",
                    plot_time_range=plot_time_range,
                    filename_in=filename,
                    output_dir=output_dir)
                plt.close(fig)

                # Line plots
                fig, ax = plt.subplots()
                if(var_name == 'tc'):
                    ax.plot(time_ws, temp_ws, '^k-', label='2.5m', markevery=500)
                    ax.plot(time_soil, temp_soil, 'vb-', label='-0.02m', markevery=500)
                if(var_name == 'wspd'):
                    y_label = "wind speed" + " (" + var_data.attrs["units"] + ")"
                    wspd_ws_rolling = pd.DataFrame(wspd_ws).rolling(120).mean().values
                    wspd_tower_rolling = pd.DataFrame(wspd_tower).rolling(120).mean().values
                    ax.plot(time_ws, wspd_ws_rolling, 'c-', label='3 m, 2 min avg', linewidth=0.5)
                    ax.plot(time_tower, wspd_tower_rolling, 'k-', label='7 m, 2 min avg', linewidth=0.5, zorder=0)
                var_data.isel(bottom_top=0, south_north=center_y, west_east=center_x).sel(Time=plot_time_range, method="nearest").plot(ax=ax, x="Time", label="WRF-Eclipse", color="orange")
                y_label = var_data.name + " (" + var_data.attrs["units"] + ")"
                save_plot(ax=ax, 
                    title='', 
                    y_label=y_label, 
                    x_label="Local Time (CDT)", 
                    var_name=var_name, 
                    plot_type_name="line_plot",
                    plot_time_range=plot_time_ranges[2],
                    filename_in=filename,
                    output_dir=output_dir)
                plt.close(fig)

        # Plot 2D values
        for var_name, var_data in two_dim_vars.items():
            # Convert to Local Time
            var_data = to_local_time(var_data)

            # Line plots
            fig, ax = plt.subplots()
            y_label = var_data.name + " (" + var_data.attrs["units"] + ")"
            if(var_name == 'swdown'):
                    ax.plot(time_ws, rad_ws, 'or-', label='measured', linewidth=0.5, markevery=500)
                    y_label = "solar radiation" + " (" + var_data.attrs["units"] + ")"
            if(var_name == 'T2'):
                    ax.plot(time_ws, temp_ws, '^k-', label='2.5m', markevery=500)
                    ax.plot(time_soil, temp_soil, 'vb-', label='-0.02m', markevery=500)
                    y_label = "temperature" + " (" + var_data.attrs["units"] + ")"
            var_data.isel(south_north=center_y, west_east=center_x).sel(Time=plot_time_range, method="nearest").plot(ax=ax, x="Time",label="WRF-Eclipse", color="orange") 
            save_plot(ax=ax, 
                title='', 
                y_label=y_label, 
                x_label="Local Time (CDT)", 
                var_name=var_name, 
                plot_type_name="line_plot",
                plot_time_range=plot_time_range,
                filename_in=filename,
                output_dir=output_dir)


if __name__ == "__main__":
    # execute only if run as a script
    main()
