import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
import numpy as np
import matplotlib.ticker as mticker
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.patches as patches
from netCDF4 import Dataset
import matplotlib.colors as mcolors
import cmaps
# 设置全局字体为  'Arial'
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.3  # 设置图框的线宽
# 创建绘图区域
fig = plt.figure(figsize=(16, 16), dpi=500)
date_range = [6*7, 6*99]
#####################################################################################################################
axe = plt.subplot(321, projection=ccrs.PlateCarree(), aspect="auto")
# 设置色阶和颜色
levels=list(np.arange(0,100000,5000))+list(np.arange(100000,400001,15000))


cmap = cmaps.WhiteBlueGreenYellowRed  # 选择 colormap
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, extend='max')


# 读取数据
ncfile = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2023.nc')
lat = ncfile.variables['lat'][:, :]
lon = ncfile.variables['lon'][:, :]
dust = ncfile.variables['dust'][date_range[0]:date_range[1], :, :]

# 绘制等值线填充图
contourf = plt.contourf(lon, lat, np.mean(dust, 0), levels, cmap=cmap, norm=norm ,extend='max')

# 标题及字体设置
plt.title('(a) Simulated Dust [mg/m$^2$]', loc='left', fontsize=20, pad=8)
for tick in axe.get_xticklabels() + axe.get_yticklabels():
    tick.set_fontname('Arial')

# 添加海岸线及范围
axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
axe.set_extent([75, 135, 30, 55], crs=ccrs.PlateCarree())

# 添加网格线
gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = False, False, False, False
gl.xlocator = mticker.FixedLocator(np.arange(80, 135, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))

# 设置坐标轴刻度及格式
axe.set_xticks(np.arange(80, 135, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())

axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelcolor='k', length=5)

# 调整刻度字体大小及参数
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tick_params(top='on', right='on', which='both')

# 读取并添加世界和中国的形状文件
shp_world = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/world/world.shp').geometries()
shp_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china/china.shp').geometries()
axe.add_geometries(shp_world, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
axe.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)

# 设置主、副刻度线
axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.tick_params(top=False, right=False)
axe.tick_params(axis="x", which="both", top=False)  # 关闭主要和次要 X 轴顶部刻度
axe.tick_params(axis="y", which="both", right=False)  # 关闭主要和次要 Y 轴右侧刻度
# 绘制颜色条
cb = fig.colorbar(contourf, ax=axe, orientation='horizontal', pad=0.1, shrink=1,aspect=22)

ticks = levels[::4]
cb.set_ticks(ticks)
cb.set_ticklabels([int(x / 1000) for x in ticks])
cb.ax.tick_params(labelsize=17)
cb.ax.tick_params(which='both', size=0)
#2##############################################################################################
# 替换原有的 subplot(322)
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import geopandas as gpd
from shapely.geometry import Point

# 读取shp区域文件
region_paths = {
    'Northwest China': '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/xibei.shp',
    'North China': '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/huabei.shp',
    'Northeast China': '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/dongbei.shp'
}

# 读取区域shp文件
region_shapes = {name: gpd.read_file(path) for name, path in region_paths.items()}

# 读取nc数据
nc_real = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2023.nc')
nc_ctl = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2005.nc')
nc_rest = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_ideal_new.nc')

lat = nc_real.variables['lat'][:, :]
lon = nc_real.variables['lon'][:, :]

# 计算绝对变化
dust_real_ctl = np.mean(nc_real.variables['dust'][date_range[0]:date_range[1],:,:], axis=0)/1000 - \
    np.mean(nc_ctl.variables['dust'][date_range[0]:date_range[1],:,:], axis=0)/1000
dust_rest_ctl = np.mean(nc_rest.variables['dust'][date_range[0]:date_range[1],:,:], axis=0)/1000 - \
    np.mean(nc_ctl.variables['dust'][date_range[0]:date_range[1],:,:], axis=0)/1000

# 计算区域平均值变化百分比
mean_real_ctl = (np.mean(dust_real_ctl) / np.mean(nc_ctl.variables['dust'][date_range[0]:date_range[1],:,:])) * 100
mean_rest_ctl = (np.mean(dust_rest_ctl) / np.mean(nc_ctl.variables['dust'][date_range[0]:date_range[1],:,:])) * 100

# 提取区域内数据
data_real_ctl, data_rest_ctl = [], []

for name, shp in region_shapes.items():
    mask = np.array([shp.contains(Point(lon[i,j], lat[i,j])).any() for i in range(lat.shape[0]) for j in range(lat.shape[1])])
    region_data_real_ctl = dust_real_ctl.flatten()[mask]
    region_data_rest_ctl = dust_rest_ctl.flatten()[mask]
    data_real_ctl.append(region_data_real_ctl)
    data_rest_ctl.append(region_data_rest_ctl)

# 在第二个子图位置绘制箱线图
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(40, 2, figure=fig)  # 10行10列的子图网格
axe = fig.add_subplot(gs[0:11, 1])  # 选择第2行的第3到第6列
# axe = plt.subplot(322)
positions = np.arange(len(region_shapes))

axe.boxplot(data_real_ctl, positions=positions-0.17, widths=0.3, patch_artist=True, 
            boxprops=dict(facecolor='blue', alpha=0.5), medianprops=dict(color='black'),
            flierprops=dict(marker='None'))
axe.boxplot(data_rest_ctl, positions=positions+0.17, widths=0.3, patch_artist=True, 
            boxprops=dict(facecolor='red', alpha=0.5), medianprops=dict(color='black'),
            flierprops=dict(marker='None'))

axe.set_xticks(positions)
axe.set_xticklabels(region_shapes.keys(), fontsize=14)
# axe.set_ylabel('Dust Difference [mg/m$^2$]', fontsize=14)
axe.grid(axis='y', linestyle=':')
import matplotlib.patches as mpatches

# 创建自定义图例句柄，确保颜色正确
legend_handles = [
    mpatches.Patch(color='blue', alpha=0.5, label='WRF_REAL - WRF_CTL'),
    mpatches.Patch(color='red', alpha=0.5, label='WRF_REST - WRF_CTL')
]

# 添加图例，确保颜色匹配
axe.legend(handles=legend_handles, fontsize=14, loc='upper right', ncol=2)
plt.title('(b) Dust Change [mg/m$^2$]', loc='left', fontsize=20, pad=8)
plt.ylim([-80,20])
plt.xlim([-0.5,2.5])
# 设置主、副刻度线
axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.tick_params(axis='both', labelsize=21)  # 调整 x 和 y 轴刻度字体大小
# 定义每个区域的背景颜色
background_colors = ['#4D7EF4', [179/255,61/255,145/255], '#0D8B43']

# 添加背景颜色
for i, color in enumerate(background_colors):
    axe.axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=0.1)

# 计算每个区域的平均变化率
region_mean_real_ctl = []
region_mean_rest_ctl = []

for name, shp in region_shapes.items():
    # 创建掩膜，筛选属于该区域的点
    mask = np.array([shp.contains(Point(lon[i, j], lat[i, j])).any() for i in range(lat.shape[0]) for j in range(lat.shape[1])])

    # 获取该区域的 dust 变化数据
    region_real_ctl = dust_real_ctl.flatten()[mask]
    region_rest_ctl = dust_rest_ctl.flatten()[mask]

    # 获取该区域的对照组（nc_ctl）平均值，确保 mask 作用在正确的数据上
    region_ctl = nc_ctl.variables['dust'][date_range[0]:date_range[1], :, :].mean(axis=0).flatten()[mask] / 1000  # 转换单位

    # 计算变化率（防止除以零）
    mean_real_ctl = (np.mean(region_real_ctl) / np.mean(region_ctl)) * 100 if np.mean(region_ctl) != 0 else np.nan
    mean_rest_ctl = (np.mean(region_rest_ctl) / np.mean(region_ctl)) * 100 if np.mean(region_ctl) != 0 else np.nan

    region_mean_real_ctl.append(mean_real_ctl)
    region_mean_rest_ctl.append(mean_rest_ctl)

# 在对应位置标注平均变化率
for i, (real_ctl, rest_ctl) in enumerate(zip(region_mean_real_ctl, region_mean_rest_ctl)):
    axe.text(i, -72, f'{real_ctl:.1f}%\n', ha='center', fontsize=22, color='blue')
    axe.text(i , -72, f'{rest_ctl:.1f}%', ha='center', fontsize=22, color='red')




# # 在图下方添加区域平均值变化的百分比
# for i, name in enumerate(region_shapes.keys()):
#     axe.text(i, axe.get_ylim()[0] - (axe.get_ylim()[1] - axe.get_ylim()[0]) * 0.1, 
#              f'{mean_real_ctl:.1f}% / {mean_rest_ctl:.1f}%', fontsize=12, ha='center')



# #2##############################################################################################
axe = plt.subplot(323, projection=ccrs.PlateCarree(), aspect="auto")

# 设置色阶和颜色
levels = np.arange(-50,51,5)
rgb = [
    [41,42,109], [39,53,126], [30,69,143], [48,101,167], [64,127,181], [81,147,195],
    [108,172,207], [138,198,221], [166,215,232], [197,229,242],[220,233,213], 
    [239,222,153], [253,205,103],[252,179,87], [245,149,65],[242,119,52], [239,81,39],
    [232,64,35], [218,44,37], [196,31,38], [167,31,45],[140,21,24],
]
rgb = np.array(rgb) / 255.0

# 读取数据
ncfile = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2023.nc')
lat = ncfile.variables['lat'][:, :]
lon = ncfile.variables['lon'][:, :]
dust = ncfile.variables['dust'][date_range[0]:date_range[1], :, :]
ncfile1 = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2005.nc')
dust1 = ncfile1.variables['dust'][date_range[0]:date_range[1], :, :]

# 绘制等值线填充图
contourf = plt.contourf(lon, lat, np.mean(dust, 0)/1000 - np.mean(dust1, 0)/1000, levels, colors=rgb, extend='both')

# 标题及字体设置
plt.title('(c) WRF_REAL - WRF_CTL: Dust [mg/m$^2$]', loc='left', fontsize=20, pad=8)
for tick in axe.get_xticklabels() + axe.get_yticklabels():
    tick.set_fontname('Arial')

# 添加海岸线及范围
axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
axe.set_extent([75, 135, 30, 55], crs=ccrs.PlateCarree())

# 添加网格线
gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = False, False, False, False
gl.xlocator = mticker.FixedLocator(np.arange(80, 135, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))

# 设置坐标轴刻度及格式
axe.set_xticks(np.arange(80, 135, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())
axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelcolor='k', length=5)

# 调整刻度字体大小及参数
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tick_params(top='on', right='on', which='both')

# 读取并添加世界和中国的形状文件
shp_world = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/world/world.shp').geometries()
shp_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china/china.shp').geometries()
axe.add_geometries(shp_world, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
axe.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
xibei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/xibei.shp').geometries()
huabei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/huabei.shp').geometries()
dongbei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/dongbei.shp').geometries()
axe.add_geometries(xibei_china, ccrs.PlateCarree(), facecolor='none', edgecolor='#4D7EF4', linewidth=2.5, linestyle='--', zorder=19)
axe.add_geometries(huabei_china, ccrs.PlateCarree(), facecolor='none', edgecolor=[179/255,61/255,145/255], linewidth=2.5, linestyle='--', zorder=20)
axe.add_geometries(dongbei_china, ccrs.PlateCarree(), facecolor='none', edgecolor='#0D8B43', linewidth=2.5, linestyle='--', zorder=19)

# 设置主、副刻度线
axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))

# 绘制颜色条
cb3 = fig.colorbar(contourf, ax=axe, orientation='horizontal', pad=0.1, shrink=1,aspect=22)
cb3.ax.tick_params(labelsize=17)
cb3.outline.set_edgecolor('black')
cb3.ax.tick_params(axis='x', colors='black')
ticks = np.arange(-50,50.1,10)
cb3.set_ticks(ticks)
cb3.set_ticklabels([int(x) for x in ticks])

# #2##############################################################################################
axe = plt.subplot(324, projection=ccrs.PlateCarree(), aspect="auto")

# 设置色阶和颜色
levels = np.arange(-5,5.1,0.5)
rgb = [
    [41,42,109], [39,53,126], [30,69,143], [48,101,167], [64,127,181], [81,147,195],
    [108,172,207], [138,198,221], [166,215,232], [197,229,242],[220,233,213], 
    [239,222,153], [253,205,103],[252,179,87], [245,149,65],[242,119,52], [239,81,39],
    [232,64,35], [218,44,37], [196,31,38], [167,31,45],[140,21,24],
]
rgb = np.array(rgb) / 255.0

# 读取数据
ncfile = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2023.nc')
lat = ncfile.variables['lat'][:, :]
lon = ncfile.variables['lon'][:, :]
dust = ncfile.variables['edust'][date_range[0]:date_range[1], :, :]
ncfile1 = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2005.nc')
dust1 = ncfile1.variables['edust'][date_range[0]:date_range[1], :, :]

# 绘制等值线填充图
contourf = plt.contourf(lon, lat, np.mean(dust, 0) - np.mean(dust1, 0), levels, colors=rgb, extend='both')
# contourf = plt.contourf(lon, lat, np.mean(dust1, 0), levels, colors=rgb, extend='both')
# 标题及字体设置
plt.title('(d) WRF_REAL - WRF_CTL: Dust Emission [µg/m$^2$/s]', loc='left', fontsize=20, pad=8)
for tick in axe.get_xticklabels() + axe.get_yticklabels():
    tick.set_fontname('Arial')

# 添加海岸线及范围
axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
axe.set_extent([85, 125, 30, 55], crs=ccrs.PlateCarree())

# 添加网格线
gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = False, False, False, False
gl.xlocator = mticker.FixedLocator(np.arange(90, 121, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))

# 设置坐标轴刻度及格式
axe.set_xticks(np.arange(90, 121, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())
axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelcolor='k', length=5)

# 调整刻度字体大小及参数
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tick_params(top='on', right='on', which='both')

# 读取并添加世界和中国的形状文件
shp_world = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/world/world.shp').geometries()
shp_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china/china.shp').geometries()
# shp_mg = shpreader.Reader('G:/shp/menggu/menggu_sheng.shp').geometries()
axe.add_geometries(shp_world, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
axe.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
# axe.add_geometries(shp_mg, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)

# 设置主、副刻度线
axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))

# # 在图上添加矩形框
# rect = patches.Rectangle((98, 42.5), 15.5, 4.5, linewidth=2, edgecolor='k', alpha=0.5, facecolor='none', linestyle='--')
# axe.add_patch(rect)

# 绘制颜色条
cb3 = fig.colorbar(contourf, ax=axe, orientation='horizontal', pad=0.1, shrink=1,aspect=22)
cb3.ax.tick_params(labelsize=17)
cb3.outline.set_edgecolor('black')
cb3.ax.tick_params(axis='x', colors='black')
ticks = np.arange(-5,5.1,1)
cb3.set_ticks(ticks)
cb3.set_ticklabels([int(x) for x in ticks])

# #2##############################################################################################
axe = plt.subplot(325, projection=ccrs.PlateCarree(), aspect="auto")

# 设置色阶和颜色
levels = np.arange(-50,51,5)
rgb = [
    [41,42,109], [39,53,126], [30,69,143], [48,101,167], [64,127,181], [81,147,195],
    [108,172,207], [138,198,221], [166,215,232], [197,229,242],[220,233,213], 
    [239,222,153], [253,205,103],[252,179,87], [245,149,65],[242,119,52], [239,81,39],
    [232,64,35], [218,44,37], [196,31,38], [167,31,45],[140,21,24],
]
rgb = np.array(rgb) / 255.0

# 读取数据
ncfile = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_ideal_new.nc')
lat = ncfile.variables['lat'][:, :]
lon = ncfile.variables['lon'][:, :]
dust = ncfile.variables['dust'][date_range[0]:date_range[1], :, :]
ncfile1 = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2005.nc')
dust1 = ncfile1.variables['dust'][date_range[0]:date_range[1], :, :]

# 绘制等值线填充图
contourf = plt.contourf(lon, lat, np.mean(dust, 0)/1000 - np.mean(dust1, 0)/1000, levels, colors=rgb, extend='both')

# 标题及字体设置
plt.title('(e) WRF_REST - WRF_CTL: Dust [mg/m$^2$]', loc='left', fontsize=20, pad=8)
for tick in axe.get_xticklabels() + axe.get_yticklabels():
    tick.set_fontname('Arial')

# 添加海岸线及范围
axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
axe.set_extent([75, 135, 30, 55], crs=ccrs.PlateCarree())

# 添加网格线
gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = False, False, False, False
gl.xlocator = mticker.FixedLocator(np.arange(80, 135, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))

# 设置坐标轴刻度及格式
axe.set_xticks(np.arange(80, 135, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())
axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelcolor='k', length=5)

# 调整刻度字体大小及参数
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tick_params(top='on', right='on', which='both')

# 读取并添加世界和中国的形状文件
shp_world = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/world/world.shp').geometries()
shp_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china/china.shp').geometries()
axe.add_geometries(shp_world, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
axe.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
xibei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/xibei.shp').geometries()
huabei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/huabei.shp').geometries()
dongbei_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china_north/dongbei.shp').geometries()
axe.add_geometries(xibei_china, ccrs.PlateCarree(), facecolor='none', edgecolor='#4D7EF4', linewidth=2.5, linestyle='--', zorder=19)
axe.add_geometries(huabei_china, ccrs.PlateCarree(), facecolor='none', edgecolor=[179/255,61/255,145/255], linewidth=2.5, linestyle='--', zorder=20)
axe.add_geometries(dongbei_china, ccrs.PlateCarree(), facecolor='none', edgecolor='#0D8B43', linewidth=2.5, linestyle='--', zorder=19)

# 设置主、副刻度线
axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))

# 绘制颜色条
cb3 = fig.colorbar(contourf, ax=axe, orientation='horizontal', pad=0.1, shrink=1,aspect=22)
cb3.ax.tick_params(labelsize=17)
cb3.outline.set_edgecolor('black')
cb3.ax.tick_params(axis='x', colors='black')
ticks = np.arange(-50,51,10)
cb3.set_ticks(ticks)
cb3.set_ticklabels([int(x) for x in ticks])

# #2##############################################################################################
axe = plt.subplot(326, projection=ccrs.PlateCarree(), aspect="auto")

# 设置色阶和颜色
levels = np.arange(-5,5.1,0.5)
rgb = [
    [41,42,109], [39,53,126], [30,69,143], [48,101,167], [64,127,181], [81,147,195],
    [108,172,207], [138,198,221], [166,215,232], [197,229,242],[220,233,213], 
    [239,222,153], [253,205,103],[252,179,87], [245,149,65],[242,119,52], [239,81,39],
    [232,64,35], [218,44,37], [196,31,38], [167,31,45],[140,21,24],
]
rgb = np.array(rgb) / 255.0

# 读取数据
ncfile = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_ideal_new.nc')
lat = ncfile.variables['lat'][:, :]
lon = ncfile.variables['lon'][:, :]
dust = ncfile.variables['edust'][date_range[0]:date_range[1], :, :]
ncfile1 = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/dust_shao04_2005.nc')
dust1 = ncfile1.variables['edust'][date_range[0]:date_range[1], :, :]

# 绘制等值线填充图
contourf = plt.contourf(lon, lat, np.mean(dust, 0) - np.mean(dust1, 0), levels, colors=rgb, extend='both')
# contourf = plt.contourf(lon, lat, np.mean(dust1, 0), levels, colors=rgb, extend='both')
# 标题及字体设置
plt.title('(f) WRF_REST - WRF_CTL: Dust Emission [µg/m$^2$/s]', loc='left', fontsize=20, pad=8)
for tick in axe.get_xticklabels() + axe.get_yticklabels():
    tick.set_fontname('Arial')

# 添加海岸线及范围
axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
axe.set_extent([85, 125, 30, 55], crs=ccrs.PlateCarree())

# 添加网格线
gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = False, False, False, False
gl.xlocator = mticker.FixedLocator(np.arange(90, 121, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))

# 设置坐标轴刻度及格式
axe.set_xticks(np.arange(90, 121, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())
axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelcolor='k', length=5)

# 调整刻度字体大小及参数
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tick_params(top='on', right='on', which='both')

# 读取并添加世界和中国的形状文件
shp_world = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/world/world.shp').geometries()
shp_china = shpreader.Reader('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/china/china.shp').geometries()
# shp_mg = shpreader.Reader('G:/shp/menggu/menggu_sheng.shp').geometries()
axe.add_geometries(shp_world, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
axe.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
# axe.add_geometries(shp_mg, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)

# 设置主、副刻度线
axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))

# # 在图上添加矩形框
# rect = patches.Rectangle((98, 42.5), 15.5, 4.5, linewidth=2, edgecolor='k', alpha=0.5, facecolor='none', linestyle='--')
# axe.add_patch(rect)

# 绘制颜色条
cb3 = fig.colorbar(contourf, ax=axe, orientation='horizontal', pad=0.1, shrink=1,aspect=22)
cb3.ax.tick_params(labelsize=17)
cb3.outline.set_edgecolor('black')
cb3.ax.tick_params(axis='x', colors='black')
ticks = np.arange(-5,5.1,1)
cb3.set_ticks(ticks)
cb3.set_ticklabels([int(x) for x in ticks])


# 最终调整与保存
# plt.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.03, right=0.95, top=0.95, wspace=0.2, hspace=0.1)
plt.savefig('figure3_Arial.png', dpi=500)
