import os
import numpy as np
from netCDF4 import Dataset

# 数据目录
datadir = r'Z:/Storage(lustre)/ProjectGroup(lzu_public)/lustre_data/EST_2/GOCART/'
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
DT = 45.0  # 积分步长: 45s

print(f'正在处理 {k} 个文件...')
for i, filename in enumerate(filelist, start=1):
    ncFilePath = os.path.join(datadir, filename)
    print(f'读取文件 {i}/{k}: {ncFilePath}')

    with Dataset(ncFilePath, 'r') as nc:
        # --- 动态计算网格面积 (m^2) ---
        # 从全局属性中读取 DX (格点间距，单位: m)
        dx = nc.DX  
        dy = nc.DY  
        grid_area = dx * dy  # 网格面积
        
        # 1. 计算 AOD (保持原样)
        extcof = nc.variables['EXTCOF55'][:] 
        ph = nc.variables['PH'][:] 
        phb = nc.variables['PHB'][:] 
        z_stag = (ph + phb) / g
        dz = np.diff(z_stag, axis=1) 
        aod_data = np.sum(extcof * (dz / 1000.0), axis=1)

        # 2. 读取并累加 Dust Load (保持原样)
        dust_sum = np.sum([nc.variables[f'DUSTLOAD_{n}'][:] for n in range(1, 6)], axis=0)

        # 3. 读取并转换 Edust (核心修改处)
        # EDUST1-5 原始单位为 kg/grid/step
        edust_raw_sum = np.sum([nc.variables[f'EDUST{n}'][:] for n in range(1, 6)], axis=0)
        
        # 换算单位为 ug/m^2/s:
        # 乘以 1e9 (kg -> ug), 除以面积 (m^2), 除以步长 (s)
        edust_flux = (edust_raw_sum * 1e9) / (grid_area * DT)

        # 4. 读取 PM10 (地面层)
        pm10_data = np.array(nc.variables['PM10'][:, 0, :, :])

        # 数据拼接逻辑 (优化内存可使用预分配数组，此处维持您的拼接方式)
        time1 = pm10_data.shape[0]
        time += time1

        if dust is None:
            dust = dust_sum
            edust = edust_flux
            pm10 = pm10_data
            aod = aod_data
        else:
            dust = np.concatenate((dust, dust_sum), axis=0)
            edust = np.concatenate((edust, edust_flux), axis=0)
            pm10 = np.concatenate((pm10, pm10_data), axis=0)
            aod = np.concatenate((aod, aod_data), axis=0)

# 读取经纬度
with Dataset(ncFilePath, 'r') as nc:
    lon = np.array(nc.variables['XLONG'][0, :, :])
    lat = np.array(nc.variables['XLAT'][0, :, :])

lat_l, lon_l = lon.shape
time_l = dust.shape[0]

output_file = './wrf_chem_dust_output_REAL_GOCART_Corrected.nc'
print(f'正在创建修正后的 NetCDF 文件: {output_file}')

with Dataset(output_file, 'w', format='NETCDF4') as nc_out:
    nc_out.createDimension('lon', lon_l)
    nc_out.createDimension('lat', lat_l)
    nc_out.createDimension('time', time_l)

    nc_out.createVariable('lon', 'f4', ('lat', 'lon'))[:, :] = lon
    nc_out.createVariable('lat', 'f4', ('lat', 'lon'))[:, :] = lat
    
    v_dust = nc_out.createVariable('dust', 'f4', ('time', 'lat', 'lon'))
    v_dust.description = "Total Dust Load (kg/m^2)"
    v_dust[:, :, :] = dust

    v_edust = nc_out.createVariable('edust', 'f4', ('time', 'lat', 'lon'))
    v_edust.description = "Total Dust Emission Flux"
    v_edust.units = "ug/m^2/s"  # 明确单位标识
    v_edust[:, :, :] = edust

    v_pm10 = nc_out.createVariable('pm10', 'f4', ('time', 'lat', 'lon'))
    v_pm10.description = "Surface PM10 Concentration"
    v_pm10[:, :, :] = pm10

    v_aod = nc_out.createVariable('aod', 'f4', ('time', 'lat', 'lon'))
    v_aod.description = "Aerosol Optical Depth at 550nm"
    v_aod[:, :, :] = aod

print('✅ 处理完成！edust 已通过面积(DX*DY)和步长(45s)归一化为 ug/m^2/s。')