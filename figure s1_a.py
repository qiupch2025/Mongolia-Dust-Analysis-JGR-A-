import cmaps
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
import numpy as np
import matplotlib.ticker as mticker
import cartopy.feature as cfeat
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from netCDF4 import Dataset
from matplotlib.colors import ListedColormap, Normalize
from scipy.stats import linregress, pearsonr
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.font_manager as fm
import maskout

# 指定 Arial 字体的完整路径
# 设置全局字体大小和样式
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.3  # 设置图框的线宽

# 创建绘图区域
fig = plt.figure(figsize=(15, 7), dpi=500)

def create_colormap():
    """创建并裁剪 colormap"""
    color_map = cmaps.MPL_terrain_r
    num_colors = 128
    original_colors = color_map(np.linspace(0, 1, num_colors))
    selected_colors = original_colors[50:120]
    return ListedColormap(selected_colors)


def plot_map_subplot(FILE, index, ndvi_layer, title):
    """绘制地图子图"""
    plt.rcParams['axes.linewidth'] = 1.3
    ax = plt.subplot(index[0],index[1],index[2], projection=ccrs.PlateCarree(), aspect="auto")
    
    # 读取数据
    dataset = nc.Dataset(FILE)
    lat, lon = dataset.variables['CLAT'][0, :, :], dataset.variables['CLONG'][0, :, :]
    ndvi = dataset.variables['ALBEDO12M'][0, ndvi_layer, :, :]/100
    
    levels = np.arange(0.05,0.35,0.01)
    color_map = create_colormap()
    norm = Normalize(vmin=0.05, vmax=0.35)
    
    contourf = plt.contourf(lon, lat, ndvi, levels, cmap=cmaps.MPL_RdBu_r, norm=norm, alpha=1, extend='both')
    # Maskout操作（如有需要）
    con_mask = maskout.shp2clip(contourf, ax, r'/home/qiupch2023/data/shp/Mongolia/Mongolia.shp')
    
    # 添加海岸线和范围
    ax.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
    ax.set_extent([86, 122, 41, 53], crs=ccrs.PlateCarree())
    
    # 添加网格线
    gridlines = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
    gridlines.top_labels = gridlines.bottom_labels = gridlines.right_labels = gridlines.left_labels = False
    gridlines.xlocator = mticker.FixedLocator(np.arange(90, 124, 15))
    gridlines.ylocator = mticker.FixedLocator(np.arange(42, 52, 3))
    
    ax.set_xticks(np.arange(90, 124, 15), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(42, 52, 3), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelcolor='k', length=5)
    
    # 添加形状文件
    shape_china = shpreader.Reader('/home/qiupch2023/data/shp/Mongolia/Mongolia.shp').geometries()
    ax.add_geometries(shape_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=1.2, zorder=1)
    
    # 标题
    plt.title(title, loc='left', fontsize=13, pad=4)
    print(title)

    # 设置主、副刻度线
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in", width=1.3, length=4, top=False, right=False)
    ax.tick_params(axis="both", which="minor", direction="in", width=1.3, length=2, top=False, right=False)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(100))

    # plt.rcParams['axes.linewidth'] = 1
    # # 绘制颜色条
    # cb3 = fig.colorbar(contourf, ax=ax, orientation='vertical', drawedges=True,pad=0.02, shrink=1,aspect=22)
    # cb3.ax.tick_params(labelsize=10)
    # cb3.outline.set_edgecolor('black')
    # # 关闭颜色条的刻度
    # cb3.ax.tick_params(left=False, right=False)
    # # 调整颜色条刻度标签与颜色条之间的距离
    # cb3.ax.yaxis.set_tick_params(pad=0)  # 将 5 替换为你需要的距离值
    # # cb3.ax.tick_params(axis='x', colors='black')
    # ticks = np.array([-0.06,-0.04,-0.02,0,0.02,0.04,0.06]) * 0.05
    # cb3.set_ticks(ticks)
    # # cb3.set_ticklabels([x * 10 for x in ticks])
    # cb3.set_ticklabels(['-0.03','-0.02','-0.01','0','0.01','0.02','0.03'])
    if index[2]==1:
        ax_inset = plt.axes([0.96, 0.15, 0.015, 0.7])  # [left, bottom, width, height] （相对 figure 的比例）
        plt.rcParams['axes.linewidth'] = 1
        # 绘制颜色条
        cb3 = fig.colorbar(contourf, cax=ax_inset, orientation='vertical', drawedges=True,pad=0.02, shrink=1,aspect=22)
        cb3.ax.tick_params(labelsize=10)
        cb3.outline.set_edgecolor('black')
        # 关闭颜色条的刻度
        cb3.ax.tick_params(left=False, right=False)
        # 调整颜色条刻度标签与颜色条之间的距离
        cb3.ax.yaxis.set_tick_params(pad=0)  # 将 5 替换为你需要的距离值
        # cb3.ax.tick_params(axis='x', colors='black')
        ticks = np.arange(0.1,0.32,0.05)
        cb3.set_ticks(ticks)
        # # cb3.set_ticklabels([x * 10 for x in ticks])
        # cb3.set_ticklabels(['-0.03','-0.02','-0.01','0','0.01','0.02','0.03'])

 
# 绘制地图子图
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01.nc', [3,4,1], 2, '(a3) Default Albedo in March')
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01_2005.nc', [3,4,2], 2, '(b3) March Albedo in WRF_CTL')
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01_2023.nc', [3,4,3], 2, '(c3) March Albedo in WRF_REAL')
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01_ideal_new.nc', [3,4,4], 2, '(d3) March Albedo in WRF_REST')
# 绘制地图子图
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01.nc', [3,4,5], 3, '(e3) Default Albedo in April')
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01_2005.nc', [3,4,6], 3, '(f3) April Albedo in WRF_CTL')
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01_2023.nc', [3,4,7], 3, '(g3) April Albedo in WRF_REAL')
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01_ideal_new.nc', [3,4,8], 3, '(h3) April Albedo in WRF_REST')
# 绘制地图子图
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01.nc', [3,4,9], 4, '(i3) Default Albedo in May')
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01_2005.nc', [3,4,10], 4, '(j3) May Albedo in WRF_CTL')
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01_2023.nc', [3,4,11], 4, '(k3) May Albedo in WRF_REAL')
plot_map_subplot('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/figure_wrf/geo_em.d01_ideal_new.nc', [3,4,12], 4, '(l3) May Albedo in WRF_REST')


plt.subplots_adjust(left=0.04, bottom=0.05, right=0.95, top=0.95, wspace=0.16, hspace=0.24)
plt.savefig('albedo_geo.png', dpi=500)
