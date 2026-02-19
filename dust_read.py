import os
import numpy as np
from netCDF4 import Dataset

datadir = 'H:/wrfout/2023/'

start_date = '2023-03-01'
end_date = '2023-05-31'

filelist = sorted([
    f for f in os.listdir(datadir) 
    if f.startswith('wrfout_d01') and start_date <= f[11:21] <= end_date
])

k = len(filelist)


dust = edust = pm10 = aod = None
time = 0
g = 9.81 


for i, filename in enumerate(filelist, start=1):
    ncFilePath = os.path.join(datadir, filename)


    with Dataset(ncFilePath, 'r') as nc:

        extcof = nc.variables['EXTCOF55'][:] 
        ph = nc.variables['PH'][:]            
        phb = nc.variables['PHB'][:]         
      
        z_stag = (ph + phb) / g
        dz = np.diff(z_stag, axis=1) 
        # AOD = sum(extcof * dz_km)
        aod_data = np.sum(extcof * (dz / 1000.0), axis=1)


        dust1 = np.array(nc.variables['DUSTLOAD_1'][:])
        dust2 = np.array(nc.variables['DUSTLOAD_2'][:])
        dust3 = np.array(nc.variables['DUSTLOAD_3'][:])
        dust4 = np.array(nc.variables['DUSTLOAD_4'][:])
        dust5 = np.array(nc.variables['DUSTLOAD_5'][:])
        dust_sum = dust1 + dust2 + dust3 + dust4 + dust5

    
        edust1 = np.array(nc.variables['EDUST1'][:])
        edust2 = np.array(nc.variables['EDUST2'][:])
        edust3 = np.array(nc.variables['EDUST3'][:])
        edust4 = np.array(nc.variables['EDUST4'][:])
        edust5 = np.array(nc.variables['EDUST5'][:])
        edust_sum = edust1 + edust2 + edust3 + edust4 + edust5

 
        pm10_data = np.array(nc.variables['PM10'][:, 0, :, :])

        time1 = dust5.shape[0]
        time += time1

   
        if dust is None:
            dust = dust_sum
            edust = edust_sum
            pm10 = pm10_data
            aod = aod_data
        else:
            dust = np.concatenate((dust, dust_sum), axis=0)
            edust = np.concatenate((edust, edust_sum), axis=0)
            pm10 = np.concatenate((pm10, pm10_data), axis=0)
            aod = np.concatenate((aod, aod_data), axis=0)



with Dataset(ncFilePath, 'r') as nc:
    lon = np.array(nc.variables['XLONG'][0, :, :])
    lat = np.array(nc.variables['XLAT'][0, :, :])

lat_l, lon_l = lon.shape
time_l = dust.shape[0]

output_file = './wrf_chem_dust_output_REAL.nc'


with Dataset(output_file, 'w', format='NETCDF4') as nc_out:

    nc_out.createDimension('lon', lon_l)
    nc_out.createDimension('lat', lat_l)
    nc_out.createDimension('time', time_l)


    nc_out.createVariable('lon', 'f4', ('lat', 'lon'))[:, :] = lon
    nc_out.createVariable('lat', 'f4', ('lat', 'lon'))[:, :] = lat
    
    v_dust = nc_out.createVariable('dust', 'f4', ('time', 'lat', 'lon'))
    v_dust.description = "Total Dust Load"
    v_dust[:, :, :] = dust

    v_edust = nc_out.createVariable('edust', 'f4', ('time', 'lat', 'lon'))
    v_edust.description = "Total Dust Emission"
    v_edust[:, :, :] = edust

    v_pm10 = nc_out.createVariable('pm10', 'f4', ('time', 'lat', 'lon'))
    v_pm10.description = "Surface PM10 Concentration"
    v_pm10[:, :, :] = pm10

    v_aod = nc_out.createVariable('aod', 'f4', ('time', 'lat', 'lon'))
    v_aod.description = "Aerosol Optical Depth at 550nm"
    v_aod[:, :, :] = aod


print('✅ 处理完成！输出文件包含：dust, edust, pm10, aod。')
