import os
import numpy as np
from netCDF4 import Dataset

# 数据目录
datadir = 'H:/wrfout/2023/'
# 定义起始和截止日期字符串
start_date = '2023-03-01'
end_date = '2023-05-31'

# 筛选逻辑：
# 1. 必须以 wrfout_d01 开头
# 2. 提取文件名中第 11 到 21 位（即 2023-03-01 长度的部分）
# 3. 判断该日期字符串是否在指定范围内
filelist = sorted([
    f for f in os.listdir(datadir) 
    if f.startswith('wrfout_d01') and start_date <= f[11:21] <= end_date
])

k = len(filelist)

# 初始化空数组
dust = edust = pm10 = aod = None
time = 0
g = 9.81  # 重力加速度，用于位势高度计算

print(f'正在处理 {k} 个文件...')
for i, filename in enumerate(filelist, start=1):
    ncFilePath = os.path.join(datadir, filename)
    print(f'读取文件 {i}/{k}: {ncFilePath}')

    with Dataset(ncFilePath, 'r') as nc:
        # 1. 计算 AOD (基于消光系数和层厚)
        extcof = nc.variables['EXTCOF55'][:]  # 单位: km^-1
        ph = nc.variables['PH'][:]             # 扰动位势
        phb = nc.variables['PHB'][:]           # 基础态位势
        # 计算层厚 dz (m)，并通过 np.diff 转换交错网格高度到层中心
        z_stag = (ph + phb) / g
        dz = np.diff(z_stag, axis=1) 
        # AOD = sum(extcof * dz_km)
        aod_data = np.sum(extcof * (dz / 1000.0), axis=1)

        # 2. 读取并累加 Dust (DUSTLOAD_1-5)
        dust1 = np.array(nc.variables['DUSTLOAD_1'][:])
        dust2 = np.array(nc.variables['DUSTLOAD_2'][:])
        dust3 = np.array(nc.variables['DUSTLOAD_3'][:])
        dust4 = np.array(nc.variables['DUSTLOAD_4'][:])
        dust5 = np.array(nc.variables['DUSTLOAD_5'][:])
        dust_sum = dust1 + dust2 + dust3 + dust4 + dust5

        # 3. 读取并累加 Edust (EDUST1-5)
        edust1 = np.array(nc.variables['EDUST1'][:])
        edust2 = np.array(nc.variables['EDUST2'][:])
        edust3 = np.array(nc.variables['EDUST3'][:])
        edust4 = np.array(nc.variables['EDUST4'][:])
        edust5 = np.array(nc.variables['EDUST5'][:])
        edust_sum = edust1 + edust2 + edust3 + edust4 + edust5

        # 4. 读取 PM10 (地面层)
        pm10_data = np.array(nc.variables['PM10'][:, 0, :, :])

        time1 = dust5.shape[0]
        time += time1

        # 数据拼接
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

    print(f'文件 {i} 处理完成，累计时间步数: {time}')

# 读取经纬度数据
with Dataset(ncFilePath, 'r') as nc:
    lon = np.array(nc.variables['XLONG'][0, :, :])
    lat = np.array(nc.variables['XLAT'][0, :, :])

lat_l, lon_l = lon.shape
time_l = dust.shape[0]

output_file = './wrf_chem_dust_output_REAL_GOCART.nc'
print(f'正在创建 NetCDF 文件: {output_file}')

# 创建 NetCDF 文件
with Dataset(output_file, 'w', format='NETCDF4') as nc_out:
    # 创建维度
    nc_out.createDimension('lon', lon_l)
    nc_out.createDimension('lat', lat_l)
    nc_out.createDimension('time', time_l)

    # 写入变量
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