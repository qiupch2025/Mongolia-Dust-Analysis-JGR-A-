# ============================================================
# 导入必要的库
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import netCDF4 as nc
from netCDF4 import Dataset

import cartopy.crs as ccrs
import cartopy.feature as cfeat
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import cmaps

# ============================================================
# 全局参数设置
# ============================================================
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 2  # 图框线宽

# ============================================================
# 创建绘图区域（上中：地图，下坐：地图，下右：半小提琴+箱线图）
# ============================================================
fig = plt.figure(figsize=(18, 18), dpi=500)
date_range = [6 * 7, 6 * 99]

# ------------------------------------------------------------
# (a) Dust from Mongolia 地图绘制
# ------------------------------------------------------------
gs = gridspec.GridSpec(2, 6, figure=fig)
axe = fig.add_subplot(gs[0, 1:5], projection=ccrs.PlateCarree(), aspect="auto")

# --- 色阶设置 ---
levels = list(np.arange(0, 51, 5)) + list(np.arange(75, 101, 25))
rgb = np.array([
    [41, 42, 109], [39, 53, 126], [30, 69, 143], [48, 101, 167], [64, 127, 181],
    [81, 147, 195], [108, 172, 207], [138, 198, 221], [166, 215, 232], [197, 229, 242],
    [220, 233, 213], [239, 222, 153], [253, 205, 103], [252, 179, 87], [245, 149, 65],
    [242, 119, 52], [239, 81, 39], [232, 64, 35], [218, 44, 37], [196, 31, 38],
    [167, 31, 45], [140, 21, 24]
]) / 255.0

# --- 读取NC数据 ---
ncfile = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2023.nc')
lat = ncfile.variables['lat'][:, :]
lon = ncfile.variables['lon'][:, :]
dust = ncfile.variables['dust'][date_range[0]:date_range[1], :, :]

ncfile1 = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/Mongolia_source/wrfout_no_mongolia/dust_shao04_no_mongolia_emission.nc')
dust1 = ncfile1.variables['dust'][date_range[0]:date_range[1], :, :]

# --- 配置颜色映射 ---
cmap = cmaps.MPL_YlOrRd
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, extend='max')

# --- 绘制主图 ---
contourf = plt.contourf(lon, lat, np.mean(dust, 0) / 1000 - np.mean(dust1, 0) / 1000,
                        levels, cmap=cmap, norm=norm, extend='max')

# --- 标题与字体 ---
plt.title('(a) Dust from Mongolia [mg/m$^2$]', loc='left', fontsize=30, pad=12)
for tick in axe.get_xticklabels() + axe.get_yticklabels():
    tick.set_fontname('Arial')

# --- 海岸线与范围 ---
axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
axe.set_extent([75, 135, 30, 55], crs=ccrs.PlateCarree())

# --- 网格线设置 ---
gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels = gl.bottom_labels = gl.right_labels = gl.left_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(80, 135, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))

# --- 坐标轴刻度 ---
axe.set_xticks(np.arange(80, 135, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())
axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelcolor='k', length=5)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.tick_params(top='on', right='on', which='both')

# --- 叠加国界和区域边界 ---
shp_world = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/world/world.shp').geometries()
shp_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china/china.shp').geometries()
axe.add_geometries(shp_world, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
axe.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)

# --- 三个子区域边界 ---
xibei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/xibei.shp').geometries()
huabei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/huabei.shp').geometries()
dongbei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/dongbei.shp').geometries()

axe.add_geometries(xibei_china, ccrs.PlateCarree(), facecolor='none', edgecolor='#4D7EF4', linewidth=2.5, linestyle='--', zorder=19)
axe.add_geometries(huabei_china, ccrs.PlateCarree(), facecolor='none', edgecolor=[179 / 255, 61 / 255, 145 / 255], linewidth=2.5, linestyle='--', zorder=20)
axe.add_geometries(dongbei_china, ccrs.PlateCarree(), facecolor='none', edgecolor='#0D8B43', linewidth=2.5, linestyle='--', zorder=19)

# --- 刻度细节 ---
axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=2, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=2, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.tick_params(axis="x", which="both", top=False)
axe.tick_params(axis="y", which="both", right=False)

# --- 颜色条 ---
cb3 = fig.colorbar(contourf, ax=axe, orientation='horizontal', pad=0.07, shrink=1, aspect=22)
cb3.ax.tick_params(labelsize=24, length=0, which='both')  # 所有刻度线都不显示
cb3.outline.set_edgecolor('black')
cb3.ax.tick_params(axis='x', colors='black')
ticks = levels[::2]
cb3.set_ticks(ticks)
cb3.set_ticklabels([int(x) for x in ticks])



# ------------------------------------------------------------
# (b) Dust from Mongolia 地图绘制
# ------------------------------------------------------------
gs = gridspec.GridSpec(2, 3, figure=fig)
axe = fig.add_subplot(gs[1, 0:2], projection=ccrs.PlateCarree(), aspect="auto")

# --- 色阶设置 ---
levels = list(np.arange(0, 21, 5)) + list(np.arange(25, 61, 5))

# --- 读取NC数据 ---
ncfile = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2023.nc')
lat = ncfile.variables['lat'][:, :]
lon = ncfile.variables['lon'][:, :]
dust = ncfile.variables['dust'][date_range[0]:date_range[1], :, :]
ncfile1 = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/Mongolia_source/wrfout_no_mongolia/dust_shao04_no_mongolia_emission.nc')
dust1 = ncfile1.variables['dust'][date_range[0]:date_range[1], :, :]

# 只保留三个shp内部的值

# 读取shp
xibei_china = list(shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/xibei.shp').geometries())
huabei_china = list(shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/huabei.shp').geometries())
dongbei_china = list(shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/dongbei.shp').geometries())

def mask_shp(shapes, lon, lat):
    mask = np.zeros(lon.shape, dtype=bool)
    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            pt = Point(lon[i, j], lat[i, j])
            if any(shape.contains(pt) for shape in shapes):
                mask[i, j] = True
    return mask

mask_xibei = mask_shp(xibei_china, lon, lat)
mask_huabei = mask_shp(huabei_china, lon, lat)
mask_dongbei = mask_shp(dongbei_china, lon, lat)
mask_all = mask_xibei | mask_huabei | mask_dongbei

# 保留mask_all内的值，其他设为np.nan
dust = np.where(mask_all, dust, np.nan)
dust1 = np.where(mask_all, dust1, np.nan)

# --- 配置颜色映射 ---
cmap = cmaps.MPL_YlOrRd
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, extend='max')

# --- 绘制主图 ---
contourf = plt.contourf(lon, lat, (np.mean(dust, 0) / 1000 - np.mean(dust1, 0) / 1000) / (np.mean(dust, 0) / 1000) * 100,
                        levels, cmap=cmap, norm=norm, extend='max')

# --- 标题与字体 ---
plt.title('(b) Contribution Rate of Dust from Mongolia [%]', loc='left', fontsize=30, pad=12)
for tick in axe.get_xticklabels() + axe.get_yticklabels():
    tick.set_fontname('Arial')

# --- 海岸线与范围 ---
axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
axe.set_extent([75, 135, 30, 55], crs=ccrs.PlateCarree())

# --- 网格线设置 ---
gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels = gl.bottom_labels = gl.right_labels = gl.left_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(80, 135, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))

# --- 坐标轴刻度 ---
axe.set_xticks(np.arange(80, 135, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())
axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelcolor='k', length=5)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.tick_params(top='on', right='on', which='both')

# --- 叠加国界和区域边界 ---
shp_world = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/world/world.shp').geometries()
shp_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china/china.shp').geometries()
axe.add_geometries(shp_world, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
axe.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)

# --- 三个子区域边界 ---
xibei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/xibei.shp').geometries()
huabei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/huabei.shp').geometries()
dongbei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/dongbei.shp').geometries()

axe.add_geometries(xibei_china, ccrs.PlateCarree(), facecolor='none', edgecolor='#4D7EF4', linewidth=2.5, linestyle='--', zorder=19)
axe.add_geometries(huabei_china, ccrs.PlateCarree(), facecolor='none', edgecolor=[179 / 255, 61 / 255, 145 / 255], linewidth=2.5, linestyle='--', zorder=20)
axe.add_geometries(dongbei_china, ccrs.PlateCarree(), facecolor='none', edgecolor='#0D8B43', linewidth=2.5, linestyle='--', zorder=19)

# --- 刻度细节 ---
axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=2, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=2, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.tick_params(axis="x", which="both", top=False)
axe.tick_params(axis="y", which="both", right=False)

# --- 颜色条 ---
cb3 = fig.colorbar(contourf, ax=axe, orientation='horizontal', pad=0.07, shrink=1, aspect=22)
cb3.ax.tick_params(labelsize=24, length=0, which='both')  # 所有刻度线都不显示
cb3.outline.set_edgecolor('black')
cb3.ax.tick_params(axis='x', colors='black')
ticks = levels[::2]
cb3.set_ticks(ticks)
cb3.set_ticklabels([int(x) for x in ticks])

# ------------------------------------------------------------
# (b) 半小提琴图 + 箱线图 (Mongolian Dust Contribution)
# ------------------------------------------------------------
gs = gridspec.GridSpec(800, 3, figure=fig)
axe = fig.add_subplot(gs[409:727, 2])

# --- 读取区域 shapefile ---
region_paths = {
    'Northwest China': '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/xibei.shp',
    'North China': '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/huabei.shp',
    'Northeast China': '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/dongbei.shp'
}
region_shapes = {name: gpd.read_file(path) for name, path in region_paths.items()}

# --- 读取 NC 文件 ---
nc_real = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2023.nc')
nc_no_mongolia_emiss = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/Mongolia_source/wrfout_no_mongolia/dust_shao04_no_mongolia_emission.nc')

lat = nc_real.variables['lat'][:, :]
lon = nc_real.variables['lon'][:, :]

# --- 计算 Dust 差值（mg/m²） ---
dust_real = np.mean(nc_real.variables['dust'][date_range[0]:date_range[1], :, :], axis=0) / 1000
dust_no_mongolia = np.mean(nc_no_mongolia_emiss.variables['dust'][date_range[0]:date_range[1], :, :], axis=0) / 1000
dust_mongolia = (dust_real - dust_no_mongolia) / dust_real * 100
dust_mongolia = np.where(dust_mongolia < 0, 0, dust_mongolia)  # 负值设为0

# --- 提取各区域数据 ---
data_mongolia = []
for name, shp in region_shapes.items():
    mask = np.array([
        shp.contains(Point(lon[i, j], lat[i, j])).any()
        for i in range(lat.shape[0])
        for j in range(lat.shape[1])
    ])
    data_mongolia.append(dust_mongolia.flatten()[mask])

# --- 转为 DataFrame ---
region_names = list(region_shapes.keys())
data_list = [(name, v) for name, data in zip(region_names, data_mongolia) for v in data]
plot_df = pd.DataFrame(data_list, columns=['Region', 'DustChange'])

# --- 半小提琴图 ---
palette = ['#4D7EF4', (179 / 255, 61 / 255, 145 / 255), '#0D8B43']
violin = sns.violinplot(x='Region', y='DustChange', data=plot_df, inner=None,
                        palette=palette, cut=0, ax=axe)
shift = 0.2
for i, coll in enumerate(violin.collections):
    for path in coll.get_paths():
        vertices = path.vertices
        x_median = np.median(vertices[:, 0])
        vertices[:, 0] = np.clip(vertices[:, 0], x_median, np.inf)
        vertices[:, 0] += shift

# --- 箱线图叠加 ---
sns.boxplot(
    x='Region', y='DustChange', data=plot_df, width=0.3, linewidth=1.3,
    boxprops={'edgecolor': 'black'}, showcaps=True,
    whiskerprops={'linewidth': 1.3}, capprops={'linewidth': 1.3},
    medianprops={'color': 'black'}, palette=palette, ax=axe,
    flierprops={'marker': 'x', 'markerfacecolor': 'black',
                'markeredgecolor': 'black', 'markersize': 3, 'linestyle': 'none'}
)

# --- 图形样式 ---
axe.set_xticks([i + 0.15 for i in range(len(region_names))])
axe.set_xticklabels([name.replace(' ', '\n') for name in region_names], fontsize=16)
for tick in axe.get_xticklabels():
    tick.set_horizontalalignment('center')

axe.grid(axis='y', linestyle=':', alpha=0.6)
axe.set_xlim([-0.3, 2.7])
axe.set_ylim([-10, 75])
axe.set_title('(c) Mongolian Dust Contribution [%]', loc='center', fontsize=30, pad=12)

# --- 坐标刻度与边框 ---
axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=2, length=7)
axe.tick_params(axis="y", which="minor", direction="out", width=2, length=3.5)
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.tick_params(axis='both', labelsize=26)
axe.tick_params(axis="x", which="minor", bottom=False)

for spine in axe.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

axe.set_xlabel('')
axe.set_ylabel('')

# --- 区域平均变化百分比 ---
mean_real, mean_no_mongolia = [], []
for name in region_names:
    shp = region_shapes[name]
    mask = np.array([
        shp.contains(Point(lon[i, j], lat[i, j])).any()
        for i in range(lat.shape[0])
        for j in range(lat.shape[1])
    ])
    region_real = dust_real.flatten()[mask]
    region_no_mongolia = dust_no_mongolia.flatten()[mask]
    mean_real.append(region_real.mean())
    mean_no_mongolia.append(region_no_mongolia.mean())

mean_real = pd.Series(mean_real, index=region_names)
mean_no_mongolia = pd.Series(mean_no_mongolia, index=region_names)
percentages = (mean_real - mean_no_mongolia) / mean_real * 100

# --- 在下方标注平均值 ---
for i, (pct, color) in enumerate(zip(percentages, palette)):
    axe.text(i + 0.15, -5, f'{pct:.1f}%', fontsize=26,
             ha='center', va='top', color=color, fontweight='bold')

# ============================================================
# 最终布局与保存
# ============================================================
plt.subplots_adjust(left=0.06, bottom=0.01, right=0.94, top=0.95, wspace=0.2, hspace=0.05)
plt.savefig('figure_mongolia_dust.png', dpi=500)
