import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, box
from shapely.prepared import prep
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 参数与路径配置
# ==========================================
# 请替换为你的真实路径
base_hysplit_path = 'G:/G盘/蒙古荒漠化沙尘工作/沙尘工作/EST_2/HYSPLIT_DATA/aa_HYSPLIT_NEW/'
pm10_dir = 'G:/G盘/蒙古荒漠化沙尘工作/aa.huifu1/airquality/站点_20230101-20231007/'
output_dir = 'G:/JGRA_DATA/' 
ndvi_path = r"Z:\Storage(lustre)\ProjectGroup(lzu_public)\lustre_data\EST_2\aaa_new\ndvi\ndvi_monthly_avg_2003-2023.nc"

# 如果输出目录不存在，自动创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

city_configs = {
    "Beijing":  "1001A",
    "Lanzhou":  "1476A",
    "Shenyang": "1099A",
    "Taiyuan":  "1081A"
}

# 设定大网格分辨率 (0.5 度)
res = 0.5
lon_range = [70, 135]
lat_range = [25, 60]

lons_arr = np.arange(lon_range[0], lon_range[1] + res, res, dtype=float)
lats_arr = np.arange(lat_range[0], lat_range[1] + res, res, dtype=float)

# ==========================================
# 2. 核心：NDVI 数据读取与重构
# ==========================================
def load_clean_ndvi(path):
    print(f">>> 正在读取 NDVI 文件: {os.path.basename(path)}")
    ds = xr.open_dataset(path)
    
    # 1. 寻找经纬度变量名
    lat_arr = None
    lon_arr = None
    for key in ds.variables:
        if 'lat' in key.lower(): lat_arr = ds[key].values
        if 'lon' in key.lower(): lon_arr = ds[key].values
            
    if lat_arr is None or lon_arr is None:
        raise ValueError("❌ 无法在文件中找到经纬度变量！")

    # 2. 提取 2023 春季数据 (3,4,5月)
    # 假设数据维度是 (time, lon, lat) 或 (time, lat, lon)
    # 假设最后一年是 2023
    raw_ndvi = ds['ndvi']
    all_vals = raw_ndvi.values 
    
    print("   -> 正在提取 2023 春季 (3,4,5月) 平均值...")
    
    # 根据数据维度进行提取 (这里假设是 [month, year, ...])
    # 如果你的 nc 只有 time 维，逻辑可能需要微调。这里沿用之前的逻辑：
    if all_vals.ndim == 4:
        # 取 index 2,3,4 (即 3,4,5月), index -1 (最后一年)
        spring_vals = all_vals[2:5, -1, :, :] 
        spring_avg = np.nanmean(spring_vals, axis=0)
    else:
        # 降级处理
        spring_avg = np.nanmean(all_vals, axis=tuple(range(all_vals.ndim - 2)))

    # 3. 转置检查 (确保是 Lat x Lon)
    target_shape = (len(lat_arr), len(lon_arr))
    if spring_avg.shape != target_shape:
        print("   -> 检测到维度转置，正在修正...")
        spring_avg = spring_avg.T

    # 4. 构建 DataArray
    clean_da = xr.DataArray(
        spring_avg,
        coords={'lat': lat_arr, 'lon': lon_arr},
        dims=('lat', 'lon')
    )
    return clean_da.sortby('lat').sortby('lon')

try:
    spring_avg_raw = load_clean_ndvi(ndvi_path)
    print("✅ NDVI 数据准备就绪。")
except Exception as e:
    print(f"❌ NDVI 读取失败: {e}")
    exit()

# ==========================================
# 3. 生成掩码 (水体剔除 + 阈值 + 湖泊剔除)
# ==========================================
n_rows = int(lats_arr.shape[0])
n_cols = int(lons_arr.shape[0])
print(f"   -> 目标网格尺寸: {n_rows} x {n_cols}")

ndvi_weight_mask = np.zeros((n_rows, n_cols))

# 提取数据加速
data_lats = spring_avg_raw.lat.values
data_lons = spring_avg_raw.lon.values
data_vals = spring_avg_raw.values

print("   -> 步骤A: 计算网格物理属性 (剔除像素级水体)...")

for i in range(n_rows):
    for j in range(n_cols):
        lat_s = float(lats_arr[i])
        lon_s = float(lons_arr[j])
        
        # 1. 提取当前网格内的所有 NDVI 像素
        mask_lat = (data_lats >= lat_s) & (data_lats < lat_s + res)
        mask_lon = (data_lons >= lon_s) & (data_lons < lon_s + res)
        
        valid_rows = data_vals[mask_lat, :]
        valid_pixels = valid_rows[:, mask_lon]
        
        # 2. 【核心】像元级水体掩膜：只保留 NDVI >= 0 的陆地像素
        valid_land_pixels = valid_pixels[valid_pixels >= 0]
        
        # 3. 计算陆地平均值并判断
        if valid_land_pixels.size > 0:
            grid_spatial_mean = np.nanmean(valid_land_pixels)
            if not np.isnan(grid_spatial_mean):
                # 陆地植被稀疏度判断
                if grid_spatial_mean < 0.12: 
                    ndvi_weight_mask[i, j] = 1.0
                else:
                    ndvi_weight_mask[i, j] = 0.0
        else:
            # 全是水体
            ndvi_weight_mask[i, j] = 0.0

print("   -> 步骤B: 应用 GIS 湖泊掩膜 (剔除贝加尔湖等大型水体)...")

# 加载 Natural Earth 湖泊数据
reader = shpreader.Reader(shpreader.natural_earth(resolution='50m', category='physical', name='lakes'))
all_lakes = list(reader.geometries())

# 空间筛选加速 (只保留研究区内的湖泊)
study_area = box(lon_range[0]-2, lat_range[0]-2, lon_range[1]+2, lat_range[1]+2)
relevant_lakes = [lake for lake in all_lakes if lake.intersects(study_area)]
lake_preps = [prep(lake) for lake in relevant_lakes]

# 遍历所有被标记为源区的点，检查是否在湖里
rows_idx, cols_idx = np.where(ndvi_weight_mask == 1)
removed_count = 0

for r, c in zip(rows_idx, cols_idx):
    # 取网格中心点坐标
    lat_p = lats_arr[r] + res/2 
    lon_p = lons_arr[c] + res/2
    p = Point(lon_p, lat_p)
    
    for lake in lake_preps:
        if lake.contains(p):
            ndvi_weight_mask[r, c] = 0.0 # 强制设为非源区
            removed_count += 1
            break 

print(f"   -> 掩膜完成。剔除湖泊误判网格数: {removed_count}")

# 验证掩膜效果
plt.figure(figsize=(8, 5))
plt.imshow(ndvi_weight_mask, origin='lower', 
           extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]], 
           cmap='Reds')
plt.colorbar(label='Is Source? (1=Yes)')
plt.title("Final Dust Source Mask")
plt.show()

# ==========================================
# 4. 准备 PM10 数据
# ==========================================
def get_pm10_lookup_table(data_dir):
    dates = pd.date_range(start="2023-03-01", end="2023-05-31", freq="D")
    all_data = pd.DataFrame()
    for date in dates:
        fpath = os.path.join(data_dir, f'china_sites_{date.strftime("%Y%m%d")}.csv')
        try:
            df = pd.read_csv(fpath, encoding='utf-8')
            sub = df.iloc[3::15].copy()
            sub.index = pd.date_range(start=date, periods=len(sub), freq='h')
            all_data = pd.concat([all_data, sub])
        except: continue
    if not all_data.empty:
        # 确保时区为 CST (Asia/Shanghai) 以匹配 HYSPLIT 转过来的时间
        all_data.index = all_data.index.tz_localize("Asia/Shanghai") if all_data.index.tz is None else all_data.index
    return all_data

print(">>> 正在加载 PM10 数据...")
pm10_master = get_pm10_lookup_table(pm10_dir)

# ==========================================
# 5. CWT 计算 (含权重)
# ==========================================
def parse_tdump(file_path):
    pts = []
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r') as f: lines = f.readlines()
        start = 0
        for i, l in enumerate(lines):
            if 'PRESSURE' in l: 
                start = i + 1
                break
        for l in lines[start:]:
            p = l.split()
            if len(p) >= 11: pts.append([float(p[9]), float(p[10])])
    except: return None
    return np.array(pts)

for city, sid in city_configs.items():
    print(f"\n--- 处理城市: {city} ---")
    traj_dir = os.path.join(base_hysplit_path, city) + '/'
    clus_file = os.path.join(traj_dir, "julei/CLUSLIST_4")
    out_nc = os.path.join(output_dir, f"{city}_PM10_CWT_Analysis_Result.nc")
    
    if not os.path.exists(clus_file): 
        print(f"⚠️ 找不到聚类文件: {clus_file}")
        continue

    sum_conc = np.zeros((n_rows, n_cols))
    sum_count = np.zeros((n_rows, n_cols))

    try:
        clus_df = pd.read_csv(clus_file, sep=r'\s+', header=None, engine='python',
                              names=["C", "N", "Y", "M", "D", "H", "I", "Path"])
        
        success_traj = 0
        for _, row in clus_df.iterrows():
            fpath = os.path.join(traj_dir, os.path.basename(str(row['Path']).strip("'")))
            
            # 时间转换: UTC -> Local
            dt_utc = datetime(int(row['Y'])+2000, int(row['M']), int(row['D']), int(row['H']))
            dt_loc = pytz.utc.localize(dt_utc).astimezone(pytz.timezone('Asia/Shanghai'))
            
            # PM10 匹配
            if dt_loc not in pm10_master.index: continue
            val = pm10_master.loc[dt_loc, sid]
            if pd.isna(val): continue
            
            # 轨迹读取
            points = parse_tdump(fpath)
            if points is None: continue
            
            success_traj += 1
            
            # 网格累加
            for lat, lon in points:
                if (lat_range[0] <= lat <= lat_range[1]) and (lon_range[0] <= lon <= lon_range[1]):
                    r_idx = int((lat - lat_range[0]) // res)
                    c_idx = int((lon - lon_range[0]) // res)
                    
                    if 0 <= r_idx < n_rows and 0 <= c_idx < n_cols:
                        sum_conc[r_idx, c_idx] += float(val)
                        sum_count[r_idx, c_idx] += 1
        
        if success_traj > 0:
            # 1. 基础 CWT
            cwt_base = np.divide(sum_conc, sum_count, out=np.zeros_like(sum_conc), where=sum_count!=0)
            
            # 2. 计算权重 (Polissar et al.)
            # 仅统计非零网格的平均经过次数
            v_counts = sum_count[sum_count > 0]
            avg = np.mean(v_counts) if len(v_counts) > 0 else 1
            
            w = np.ones_like(sum_count)
            w[sum_count <= 3*avg] = 0.70
            w[sum_count <= 2*avg] = 0.42
            w[sum_count <= avg] = 0.17
            
            cwt_weighted = cwt_base * w
            
            # 3. 应用 NDVI 物理约束
            cwt_final = cwt_weighted * ndvi_weight_mask 
            
            # 4. 保存
            ds = xr.Dataset(
                {
                    "CWT_Final": (["lat", "lon"], cwt_final.astype(np.float32)),
                    "CWT_Original": (["lat", "lon"], cwt_weighted.astype(np.float32)),
                    "Dust_Source_Mask": (["lat", "lon"], ndvi_weight_mask.astype(np.float32)),
                    "Trajectory_Count": (["lat", "lon"], sum_count.astype(np.int32))
                },
                coords={"lat": lats_arr, "lon": lons_arr}
            )
            ds.to_netcdf(out_nc)
            print(f"✅ {city} 计算完成，已保存至 {out_nc}")
        else:
            print(f"⚠️ {city} 有效轨迹数为 0")
            
    except Exception as e:
        print(f"⚠️ {city} 处理过程中出错: {e}")

print("\n>>> 所有任务完成。")

# import os
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cartopy.io.shapereader as shpreader
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# from shapely.geometry import Point
# from shapely.prepared import prep
# import warnings

# warnings.filterwarnings("ignore")

# # ==========================================
# # 1. 配置参数
# # ==========================================
# input_dir = 'G:/JGRA_DATA/'
# # 城市顺序：北京、沈阳、兰州
# cities = ["Beijing", "Shenyang", "Lanzhou"] 
# file_suffix = "_PM10_CWT_Analysis_Result.nc"

# extent = [70, 135, 25, 60]
# city_coords = {
#     "Beijing":  (116.40, 39.90),
#     "Shenyang": (123.43, 41.80),
#     "Lanzhou":  (103.83, 36.06)
# }

# # 色标范围
# vmin = 0
# vmax = 200 

# # ==========================================
# # 2. 准备国界数据
# # ==========================================
# print(">>> 正在准备地理数据...")
# resolution = '110m'
# shpfilename = shpreader.natural_earth(resolution=resolution, category='cultural', name='admin_0_countries')
# reader = shpreader.Reader(shpfilename)
# countries = list(reader.records())

# mongolia_geom, china_geom = None, None
# for c in countries:
#     name = c.attributes['NAME']
#     if name == 'Mongolia': mongolia_geom = c.geometry
#     elif name == 'China': china_geom = c.geometry

# if not mongolia_geom or not china_geom:
#     print("❌ 国界数据缺失")
#     exit()

# mongolia_prep = prep(mongolia_geom)
# china_prep = prep(china_geom)

# def calculate_contribution(ds):
#     """计算贡献值 (包含0值以便累加)"""
#     lats, lons = ds.lat.values, ds.lon.values
#     cwt = ds['CWT_Final'].values
#     cwt = np.nan_to_num(cwt)
    
#     sum_m, sum_c, sum_total = 0, 0, np.sum(cwt)
#     rows, cols = np.where(cwt > 0)
#     for r, c in zip(rows, cols):
#         lat, lon = lats[r], lons[c]
#         p = Point(lon, lat)
#         val = cwt[r, c]
#         if mongolia_prep.contains(p):
#             sum_m += val
#         elif china_prep.contains(p):
#             sum_c += val
            
#     sum_o = sum_total - sum_m - sum_c
#     if sum_o < 0: sum_o = 0
#     return sum_m, sum_c, sum_o, sum_total

# # ==========================================
# # 3. 绘图主程序
# # ==========================================
# def main():
#     print(">>> 开始绘制 2x2 布局图...")
    
#     # 画布大小 (正方形布局，稍微宽一点)
#     fig = plt.figure(figsize=(14, 12), dpi=300)
    
#     # 存储统计数据
#     stats_data = {"Mongolia": [], "China": [], "Others": []}
    
#     # 颜色设置
#     cmap = plt.cm.get_cmap('jet').copy()
#     cmap.set_bad('white', alpha=0)
    
#     # 定义子图位置编号 (1, 2, 3 是地图)
#     # subplot_indices: 
#     # 1 (Top-Left), 2 (Top-Right)
#     # 3 (Bottom-Left), 4 (Bottom-Right -> Bar Chart)
    
#     mesh = None # 用于色标
    
#     # --- 循环绘制 3 个地图 (位置 1, 2, 3) ---
#     for i, city in enumerate(cities):
#         # 注意：add_subplot(2, 2, index) index 从 1 开始
#         idx = i + 1 
#         ax = fig.add_subplot(2, 2, idx, projection=ccrs.PlateCarree())
        
#         fpath = os.path.join(input_dir, f"{city}{file_suffix}")
#         if not os.path.exists(fpath): continue
        
#         ds = xr.open_dataset(fpath)
        
#         # 1. 计算贡献
#         m, c, o, t = calculate_contribution(ds)
#         stats_data["Mongolia"].append(m/t*100)
#         stats_data["China"].append(c/t*100)
#         stats_data["Others"].append(o/t*100)
        
#         # 2. 准备绘图数据 (掩膜 0 值)
#         plot_cwt = ds['CWT_Final'].where(ds['CWT_Final'] > 0)
        
#         # 3. 画热力图
#         mesh = plot_cwt.plot.pcolormesh(
#             ax=ax, transform=ccrs.PlateCarree(),
#             cmap=cmap, vmin=vmin, vmax=vmax,
#             add_colorbar=False, rasterized=True
#         )
        
#         # 4. 画 NDVI < 0.12 红色轮廓
#         ax.contour(
#             ds.lon, ds.lat, ds['Dust_Source_Mask'], 
#             levels=[0.5], colors='red', linewidths=0.9, 
#             transform=ccrs.PlateCarree()
#         )
        
#         # 5. 地图装饰
#         ax.set_extent(extent, crs=ccrs.PlateCarree())
#         ax.add_feature(cfeature.LAND, facecolor='white')
#         ax.add_feature(cfeature.COASTLINE, lw=0.6)
#         ax.add_feature(cfeature.BORDERS, linestyle='-', lw=0.4, alpha=0.6)
        
#         # 标记城市
#         cx, cy = city_coords[city]
#         ax.plot(cx, cy, marker='*', color='k', ms=14, mec='yellow', mew=0.8, transform=ccrs.PlateCarree(), zorder=10)
        
#         # 坐标轴
#         ax.set_xticks(np.arange(70, 140, 15), crs=ccrs.PlateCarree())
#         ax.set_yticks(np.arange(30, 65, 10), crs=ccrs.PlateCarree())
#         ax.xaxis.set_major_formatter(LongitudeFormatter())
#         ax.yaxis.set_major_formatter(LatitudeFormatter())
#         ax.tick_params(labelsize=10)
        
#         # 标题 (a) (b) (c)
#         letter = chr(97 + i)
#         ax.set_title(f"({letter}) {city}", loc='left', fontsize=14, fontweight='bold', pad=5)
#         ax.set_xlabel("")
#         ax.set_ylabel("")

#     # --- 添加地图统一色标 (水平，放在整个图的底部中央) ---
#     # [left, bottom, width, height]
#     cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.02]) 
#     cb = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal', extend='max')
#     cb.set_label('Weighted CWT Concentration ($\mu g/m^3$)', fontsize=12)

#     # ==========================================
#     # 4. 右下角：竖向堆叠柱状图 (普通坐标系)
#     # ==========================================
#     ax_bar = fig.add_subplot(2, 2, 4) # 第 4 个位置
    
#     # 柱状图配色 (Nature 风格)
#     c_m = '#D62728' # 蒙古 (红)
#     c_c = '#FF7F0E' # 中国 (橙)
#     c_o = '#F0F0F0' # 其他 (灰)
    
#     # X轴位置
#     x_pos = np.arange(len(cities))
#     bar_width = 0.5
    
#     # 绘制竖向堆叠柱 (Bottom 参数关键)
#     # 1. 底部: 蒙古
#     p1 = ax_bar.bar(x_pos, stats_data["Mongolia"], width=bar_width, color=c_m, label='Mongolia', zorder=3)
    
#     # 2. 中间: 中国 (bottom=蒙古)
#     p2 = ax_bar.bar(x_pos, stats_data["China"], width=bar_width, bottom=stats_data["Mongolia"], 
#                    color=c_c, label='China (Domestic)', zorder=3)
    
#     # 3. 顶部: 其他 (bottom=蒙古+中国)
#     bottom_others = [m+c for m,c in zip(stats_data["Mongolia"], stats_data["China"])]
#     p3 = ax_bar.bar(x_pos, stats_data["Others"], width=bar_width, bottom=bottom_others, 
#                    color=c_o, edgecolor='gray', label='Others', zorder=3)
    
#     # 数值标签 (竖着画时，文字要在柱子中间)
#     def add_labels(stats, bottom_vals, color='white'):
#         for i, val in enumerate(stats):
#             if val > 5: # 数值太小不显示
#                 height = bottom_vals[i] + val/2 if bottom_vals else val/2
#                 ax_bar.text(i, height, f"{val:.1f}%", ha='center', va='center', 
#                            color=color, fontweight='bold', fontsize=10)

#     add_labels(stats_data["Mongolia"], None, 'white')
#     add_labels(stats_data["China"], stats_data["Mongolia"], 'white')
#     # add_labels(stats_data["Others"], bottom_others, '#333333') # 其他部分通常不需要标，除非很重要

#     # 美化柱状图
#     ax_bar.set_ylim(0, 100)
#     ax_bar.set_xticks(x_pos)
#     ax_bar.set_xticklabels(cities, fontsize=12, fontweight='bold')
#     ax_bar.set_ylabel('Contribution Percentage (%)', fontsize=12, fontweight='bold')
    
#     # 标题 (d)
#     ax_bar.set_title("(d) Source Contribution", loc='left', fontsize=14, fontweight='bold')
    
#     # 网格线与去边框
#     ax_bar.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
#     ax_bar.spines['top'].set_visible(False)
#     ax_bar.spines['right'].set_visible(False)
    
#     # 图例 (放在图内部或上方)
#     ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=10)

#     # 调整整体布局
#     plt.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.12)

#     # 保存
#     save_path = os.path.join(input_dir, "Figure3_2x2_Final.png")
#     plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     print(f"✅ 2x2 布局图已保存: {save_path}")
#     plt.show()

# if __name__ == "__main__":
#     main()

# import os
# import numpy as np
# import xarray as xr
# import rioxarray  # 核心库：连接 xarray 和 rasterio
# import geopandas as gpd
# import pandas as pd
# from shapely.geometry import Point, shape
# from rasterio.features import shapes
# import warnings

# warnings.filterwarnings("ignore")

# # ==========================================
# # 1. 配置路径
# # ==========================================
# input_dir = 'G:/JGRA_DATA/' 
# cities = ["Beijing", "Shenyang", "Lanzhou"]
# file_suffix = "_PM10_CWT_Analysis_Result.nc"

# # 城市坐标
# city_coords = {
#     "Beijing":  (116.40, 39.90),
#     "Shenyang": (123.43, 41.80),
#     "Lanzhou":  (103.83, 36.06)
# }

# # 输出文件夹
# output_gis_dir = os.path.join(input_dir, "GIS_Data_Output")
# if not os.path.exists(output_gis_dir):
#     os.makedirs(output_gis_dir)

# # ==========================================
# # 2. 功能函数：将 xarray 转为 GeoTIFF
# # ==========================================
# def export_cwt_to_tiff(city_name):
#     nc_path = os.path.join(input_dir, f"{city_name}{file_suffix}")
#     if not os.path.exists(nc_path):
#         print(f"⚠️ 找不到文件: {nc_path}")
#         return None

#     try:
#         # 读取数据
#         ds = xr.open_dataset(nc_path)
        
#         # 提取 CWT_Final
#         da = ds['CWT_Final']
        
#         # -------------------------------------------------------
#         # 【核心修复】：强制改名为 x, y，彻底解决 rioxarray 找不到维度的问题
#         # -------------------------------------------------------
#         if 'lon' in da.dims and 'lat' in da.dims:
#             da = da.rename({'lon': 'x', 'lat': 'y'})
        
#         # 赋予地理参考 (WGS84)
#         da.rio.write_crs("EPSG:4326", inplace=True)
        
#         # 设置 NoData 值 (将 0 或 NaN 设为透明)
#         da = da.where(da > 0) # 确保 0 变成 NaN
#         da.rio.write_nodata(np.nan, inplace=True)
        
#         # 保存路径
#         tif_path = os.path.join(output_gis_dir, f"{city_name}_CWT_Final.tif")
        
#         # 导出
#         da.rio.to_raster(tif_path, compress='LZW') # LZW压缩减小体积
#         print(f"✅ [TIFF] {city_name} 已导出: {tif_path}")
        
#         return ds # 返回 dataset 供下一步提取掩码用
        
#     except Exception as e:
#         print(f"❌ {city_name} TIFF 导出失败: {e}")
#         return None

# # ==========================================
# # 3. 功能函数：将 Mask 转为 Shapefile
# # ==========================================
# def export_mask_to_shp(ds_sample):
#     print(">>> 正在生成潜在源区 Shapefile...")
    
#     try:
#         # 提取掩码 (0/1 矩阵)
#         mask_da = ds_sample['Dust_Source_Mask']
        
#         # -------------------------------------------------------
#         # 【核心修复】：这里也要改名为 x, y 才能正确获取 transform
#         # -------------------------------------------------------
#         if 'lon' in mask_da.dims and 'lat' in mask_da.dims:
#             mask_da = mask_da.rename({'lon': 'x', 'lat': 'y'})
        
#         mask_da.rio.write_crs("EPSG:4326", inplace=True)
        
#         # 转换为 numpy 数组 (必须是 float32 或 int)
#         mask_arr = mask_da.values.astype('float32')
        
#         # 获取仿射变换参数 (用于将数组索引转为经纬度)
#         transform = mask_da.rio.transform()
        
#         # 使用 rasterio.features.shapes 进行矢量化
#         # 这一步会把值为 1 的所有网格连成多边形
#         results = (
#             {'properties': {'value': v}, 'geometry': s}
#             for i, (s, v) in enumerate(shapes(mask_arr, mask=None, transform=transform))
#             if v == 1  # 只保留值为 1 (源区) 的部分
#         )
        
#         # 构建 GeoDataFrame
#         geoms = list(results)
#         if not geoms:
#             print("⚠️ 警告：掩码中没有检测到源区 (值为1的区域)")
#             return

#         polygons = [shape(g['geometry']) for g in geoms]
#         gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")
        
#         # 保存
#         shp_path = os.path.join(output_gis_dir, "Potential_Dust_Source_Area.shp")
#         gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
#         print(f"✅ [SHP] 潜在源区矢量已导出: {shp_path}")
        
#     except Exception as e:
#         print(f"❌ SHP 导出失败: {e}")

# # ==========================================
# # 4. 功能函数：将城市点 **分别** 转为 Shapefile (已修改)
# # ==========================================
# def export_cities_to_shp():
#     print(">>> 正在生成各城市单独的 Shapefile...")
    
#     try:
#         for city, (lon, lat) in city_coords.items():
#             # 1. 创建单点数据
#             names = [city]
#             geometry = [Point(lon, lat)]
            
#             # 2. 创建 GeoDataFrame
#             df = pd.DataFrame({'City_Name': names})
#             gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            
#             # 3. 单独保存
#             shp_name = f"{city}_Location.shp"
#             shp_path = os.path.join(output_gis_dir, shp_name)
            
#             gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
#             print(f"✅ [SHP] {city} 站点位置已导出: {shp_path}")
            
#     except Exception as e:
#         print(f"❌ 城市点导出失败: {e}")

# # ==========================================
# # 5. 主程序
# # ==========================================
# def main():
#     print(f"STARTING GIS EXPORT -> {output_gis_dir}")
    
#     # 1. 导出三个 TIFF 并获取一个样本 DS (用于做 Mask)
#     sample_ds = None
#     for city in cities:
#         ds = export_cwt_to_tiff(city)
#         if sample_ds is None and ds is not None: 
#             sample_ds = ds
            
#     # 2. 导出潜在源区 SHP (只需要用其中一个文件的 Mask 即可，因为物理约束是一样的)
#     if sample_ds:
#         export_mask_to_shp(sample_ds)
#     else:
#         print("⚠️ 未能读取到有效的 NetCDF 文件，跳过 Mask 导出。")
        
#     # 3. 导出城市点 SHP (分别导出)
#     export_cities_to_shp()
    
#     print("\n🎉 所有 GIS 数据转换完成！")
#     print(f"文件位置: {output_gis_dir}")

# if __name__ == "__main__":
#     main()