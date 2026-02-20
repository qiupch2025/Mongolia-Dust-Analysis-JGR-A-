import xarray as xr
import geopandas as gpd
import regionmask
import pandas as pd
import numpy as np
import os
import glob

# --- 配置参数 ---
data_dir = "G:/JGRA_DATA/merra2_data3"
output_dir = "G:/JGRA_DATA/merra2_data3/processed_data"
if not os.path.exists(output_dir): os.makedirs(output_dir)

receptor_config = {
    "NEC": "G:/G盘/shp/north_china/dongbei.shp",
    "NC": "G:/G盘/shp/north_china/huabei.shp",
    "NWC": "G:/G盘/shp/north_china/xibei.shp"
}
source_config = {
    "Mongolia": "G:/G盘/shp/menggu/menggu.shp",
    "Xinjiang": "G:/JGRA_DATA/shp/xinjiang.shp",
    "Inner_Mongolia": "G:/G盘/shp/Inner_Mongolia/Inner_Mongolia.shp",
}

years = range(1980, 2026)
spring_months = ["03", "04", "05"]
# 缩放因子：浓度 mg/m3 -> ug/m3 (1e6), 排放 kg/m2/s -> ug/m2/s (1e9)
DUCMASS_scale, DUEM_scale = 1e6, 1e9

def extract_ts(da, shp_path):
    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    mask = regionmask.mask_geopandas(gdf, da.lon, da.lat)
    return da.where(~np.isnan(mask)).mean(dim=['lat', 'lon'])

# --- 核心处理循环 ---
monthly_records = []
all_emiss_maps = [] # 用于计算多年平均空间分布

print(">>> 正在提取 1980-2025 数据...")

for year in years:
    year_emiss_list = []
    for m in spring_months:
        f_aer = glob.glob(os.path.join(data_dir, f"*aer*{year}{m}*.nc*"))
        f_adg = glob.glob(os.path.join(data_dir, f"*adg*{year}{m}*.nc*"))
        
        if f_aer and f_adg:
            with xr.open_dataset(f_aer[0]) as ds_aer, xr.open_dataset(f_adg[0]) as ds_adg:
                # 1. 空间场准备
                da_duc = ds_aer['DUCMASS'].mean(dim='time') * DUCMASS_scale
                bins = [f"DUEM00{i}" for i in range(1, 6)]
                da_emiss = sum(ds_adg[b] for b in ds_adg.data_vars if b in bins).mean(dim='time') * DUEM_scale
                
                # 2. 提取区域平均值 (用于时间序列和回归)
                curr_time = pd.to_datetime(f"{year}-{m}-01")
                row = {'time': curr_time}
                for name, shp in source_config.items():
                    row[f"src_{name}"] = float(extract_ts(da_emiss, shp))
                for name, shp in receptor_config.items():
                    row[f"rec_{name}"] = float(extract_ts(da_duc, shp))
                
                monthly_records.append(row)
                year_emiss_list.append(da_emiss)
                
    if year_emiss_list:
        all_emiss_maps.append(xr.concat(year_emiss_list, dim='month').mean(dim='month'))
    print(f"Year {year} processed.")

# 1. 保存时间序列 CSV
df_final = pd.DataFrame(monthly_records).set_index('time')
df_final.to_csv(f"{output_dir}/merra2_dust_ts_1980_2025.csv")

# 2. 保存气候态空间场 NC
clim_emiss = xr.concat(all_emiss_maps, dim='year').mean(dim='year')
clim_emiss.name = 'dust_emiss'
clim_emiss.to_netcdf(f"{output_dir}/climatology_emiss.nc")

print(f">>> 处理完成！文件存放在: {output_dir}")