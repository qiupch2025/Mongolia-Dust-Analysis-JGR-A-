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

# 指定 Arial 字体的完整路径
font_path = "D:/ProgramData/anaconda3/envs/py310/Lib/site-packages/geemap/data/fonts/Arial.ttf"

# 使用 FontProperties 加载字体
Arial_font = fm.FontProperties(fname=font_path)

# 设置全局字体（必须使用 fontproperties）
mpl.rcParams['font.family'] = Arial_font.get_name()  # 这里不能直接使用 'Arial'
plt.rcParams['axes.linewidth'] = 1.3

def create_colormap(color_map):
    """创建并裁剪 colormap"""
    # color_map = cmaps.temp_diff_18lev_r
    num_colors = 256
    original_colors = color_map(np.linspace(0, 1, num_colors))
    selected_colors = original_colors[:]
    return ListedColormap(selected_colors)


def plot_map_subplot(m_file, im_file, var_name,color_map, index, title, fig,
                     shp_path_im, shp_path_mn, extent=[83, 128, 37, 54]):
    """绘制地图子图（两个 contourf 使用各自 lon/lat）"""
    ax = fig.add_subplot(index, projection=ccrs.PlateCarree(), aspect="auto")

    # 读取 Inner Mongolia 数据
    ds_im = nc.Dataset(im_file)
    lat_im = ds_im.variables['latitude'][:, :]
    lon_im = ds_im.variables['longitude'][:, :]
    data_im = ds_im.variables[var_name][:, :]

    # 读取 Mongolia 数据
    ds_m = nc.Dataset(m_file)
    lat_m = ds_m.variables['latitude'][:, :]
    lon_m = ds_m.variables['longitude'][:, :]
    data_m = ds_m.variables[var_name][:, :]

    # 设置色阶与颜色图
    levels = np.array([-1, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 1]) * 0.05

    norm = Normalize(vmin=-0.0035, vmax=0.0035)

    # 分别绘制两个区域的数据
    contourf1 = ax.contourf(lon_im, lat_im, data_im, levels, cmap=color_map, norm=norm, alpha=1)
    contourf2 = ax.contourf(lon_m, lat_m, data_m, levels, cmap=color_map, norm=norm, alpha=1)

    # 添加地理要素与范围
    ax.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0.5, color='k')
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 经纬度网格
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, linestyle=':')
    gl.top_labels = gl.right_labels = gl.left_labels = gl.bottom_labels = False
    ax.set_xticks(np.arange(85, extent[1] + 1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(extent[2], extent[3] + 1, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelcolor='k', length=5)

    # 添加行政边界
    shape_im = shpreader.Reader(shp_path_im).geometries()
    shape_mn = shpreader.Reader(shp_path_mn).geometries()
    ax.add_geometries(shape_im, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=1.2)
    ax.add_geometries(shape_mn, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=1.2)

    # 标题
    ax.set_title(title, loc='left', fontsize=13)

    # 添加颜色条（使用 im 区域数据为主）
    cb = fig.colorbar(contourf1, ax=ax, orientation='vertical', pad=0.02, shrink=1, aspect=22, drawedges=True)
    cb.outline.set_edgecolor('black')
    cb.ax.tick_params(labelsize=12, left=False, right=False, pad=0)
    ticks = np.array([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06]) * 0.05
    cb.set_ticks(ticks)
    cb.set_ticklabels(['-0.03', '-0.02', '-0.01', '0', '0.01', '0.02', '0.03'])
    # 设置主、副刻度线
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="out", width=1.3, length=4, top=False, right=False)
    ax.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3, top=False, right=False)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(100))
# --- 保持图不变，仅调整图例顺序 ---
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patches as mpatches

    # 1. 容器创建
    ax_ins = inset_axes(ax, width="35%", height="26%", loc='lower left', 
                        bbox_to_anchor=(0.04, -0.03, 1, 1), bbox_transform=ax.transAxes)

    # 变量匹配
    p_var_mapping = {'ndvi_trend': 'ndvi_pvalue', 'lai_trend': 'lai_pvalue', 'albedo_trend': 'albedo_pvalue'}
    curr_p_var = p_var_mapping.get(var_name, var_name + '_pvalue')

    def get_stats(data_arr, ds_obj):
        d = data_arr.flatten()
        p = ds_obj.variables[curr_p_var][:].flatten()
        mask = ~np.isnan(d)
        d, p = d[mask], p[mask]
        total = len(d) if len(d) > 0 else 1
        return d, p, total

    d_im, p_im, total_im = get_stats(data_im, ds_im)
    d_m, p_m, total_m = get_stats(data_m, ds_m)

    # 2. 改善趋势计算 (Albedo 逻辑反转)
    is_albedo = 'Albedo' in title
    c_imp = "#2E7D32" 

    if is_albedo:
        v_tot_im, v_sig_im = np.sum(d_im <= 0)/total_im*100, np.sum((d_im <= 0) & (p_im < 0.05))/total_im*100
        v_tot_m, v_sig_m = np.sum(d_m <= 0)/total_m*100, np.sum((d_m <= 0) & (p_m < 0.05))/total_m*100
    else:
        v_tot_im, v_sig_im = np.sum(d_im > 0)/total_im*100, np.sum((d_im > 0) & (p_im < 0.05))/total_im*100
        v_tot_m, v_sig_m = np.sum(d_m > 0)/total_m*100, np.sum((d_m > 0) & (p_m < 0.05))/total_m*100

    # 3. 绘图执行 (保持：显著在上，总数在下)
    y_pos = [1, 0] 
    bar_h = 0.38 

    def plot_improvement_group(y, v_tot, v_sig):
        # 显著改善 (上方)
        ax_ins.barh(y + bar_h/2, v_sig, bar_h, color=c_imp, hatch='////', edgecolor='white', lw=0.5)
        ax_ins.text(v_sig + 2, y + bar_h/2, f'{v_sig:.0f}%', va='center', fontsize=12, color=c_imp, fontweight='bold')
        
        # 总改善 (下方)
        ax_ins.barh(y - bar_h/2, v_tot, bar_h, color=c_imp, alpha=0.5)
        ax_ins.text(v_tot + 2, y - bar_h/2, f'{v_tot:.0f}%', va='center', fontsize=12, color=c_imp, alpha=0.7, fontweight='bold')

    plot_improvement_group(y_pos[0], v_tot_m, v_sig_m)
    plot_improvement_group(y_pos[1], v_tot_im, v_sig_im)

    # 4. 坐标轴修饰
    ax_ins.set_yticks(y_pos)
    ax_ins.set_yticklabels([]) 
    ax_ins.text(-7, y_pos[0] + 0.05, 'MN', fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
    ax_ins.text(-7, y_pos[1] + 0.05, 'IM', fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
    ax_ins.set_xlim(0, 155)
    ax_ins.set_xticks([])
    ax_ins.text(-0.15, 1.85, 'Area (%)', fontsize=11, fontweight='bold', transform=ax_ins.transAxes)

    # 5. 调整图例顺序：Improved 在上，Sig. Improved 在下
    handle_tot = mpatches.Patch(color=c_imp, alpha=0.5, label='Improved')
    handle_sig = mpatches.Patch(facecolor=c_imp, hatch='////', label='Sig. Improved', edgecolor='white')

    # 注意这里的 handles 顺序：tot 在前，sig 在后 (对应图例中从上到下的顺序)
    leg = ax_ins.legend(handles=[handle_tot, handle_sig], 
                        loc='upper left', bbox_to_anchor=(-0.24, 1.8), 
                        ncol=1, frameon=False,
                        handlelength=1.5, handleheight=0.8,
                        labelspacing=0.5, handletextpad=0.4,
                        prop={'size': 10, 'weight': 'bold'})

    # 6. 清理
    for s in ['top', 'right', 'left', 'bottom']: ax_ins.spines[s].set_visible(False)
    ax_ins.tick_params(left=False, bottom=False)
    ax_ins.patch.set_alpha(0)

def plot_trend_subplot(position, y_lim, text, ticknum, title, fig,
                       ncfile_path_im, ncfile_path_mn, var_name_line):
    """绘制趋势图子图（双Y轴，显著性标注分两行，坐标轴颜色一致）"""
    axe = fig.add_subplot(position)
    axe2 = axe.twinx()

    years = np.arange(2005, 2024)
    ds_im = nc.Dataset(ncfile_path_im)
    ds_mn = nc.Dataset(ncfile_path_mn)
    data_im = ds_im.variables[var_name_line][:]
    data_mn = ds_mn.variables[var_name_line][:]

    # 拟合
    slope_im, intercept_im, r_im, p_im, _ = linregress(years, data_im)
    fit_im = slope_im * years + intercept_im
    trend_im = slope_im / fit_im[0] * 100

    slope_mn, intercept_mn, r_mn, p_mn, _ = linregress(years, data_mn)
    fit_mn = slope_mn * years + intercept_mn
    trend_mn = slope_mn / fit_mn[0] * 100

    # 绘图
    axe.plot(years, data_im, 'o', color='g', markersize=6)
    axe.plot(years, fit_im, 'g--')
    axe2.plot(years, data_mn, 'v', color='r', markersize=6)
    axe2.plot(years, fit_mn, 'r--')

    # 显著性星号
    sign_im = "**" if p_im < 0.01 else "*" if p_im < 0.05 else ""
    sign_mn = "**" if p_mn < 0.01 else "*" if p_mn < 0.05 else ""

    # 文本位置
    x1, y1 = 2005, y_lim[0] + text[0] * (y_lim[1] - y_lim[0])
    x2, y2 = 2015, y_lim[0] + text[1] * (y_lim[1] - y_lim[0])
    y_offset = 0.07 * (y_lim[1] - y_lim[0])

    # 标注 R² 和 Rate（两行）
    axe.text(x1, y1, f"$R^2={r_im**2:.2f}{sign_im}$", fontsize=12, color='g')
    axe.text(x1, y1 - y_offset, f"$\\mathrm{{Rate}}:~{trend_im:.2f}\\%~a^{{-1}}$", fontsize=12, color='g')
    axe.text(x2, y2, f"$R^2={r_mn**2:.2f}{sign_mn}$", fontsize=12, color='r')
    axe.text(x2, y2 - y_offset, f"$\\mathrm{{Rate}}:~{trend_mn:.2f}\\%~a^{{-1}}$", fontsize=12, color='r')

    # 坐标与样式设置
    axe.set_xlim(2004.5, 2023.5)
    axe.set_ylim(y_lim[0], y_lim[1])
    axe2.set_ylim(y_lim[2], y_lim[3])
    axe.set_title(title, loc='left', fontsize=13)
    axe.set_xlabel("Year", fontsize=13)
    axe.set_xticks(np.arange(2005, 2024, 6))
    axe.set_yticks(np.linspace(y_lim[0], y_lim[1], ticknum))
    axe2.set_yticks(np.linspace(y_lim[2], y_lim[3], ticknum))
    axe.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

    # 设置坐标轴颜色
    axe.tick_params(axis='y', labelcolor='g')
    axe2.tick_params(axis='y', labelcolor='r')
    # axe.spines['left'].set_color('g')
    # axe2.spines['right'].set_color('r')
    axe.set_ylabel("Inner Mongolia", fontsize=13, color='g')
    axe2.set_ylabel("Mongolia", fontsize=13, color='r',rotation=270,labelpad=12)
    axe.minorticks_on()
    axe2.minorticks_on()
    # 仅显示下边和左边的刻度线
    axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=4, top=False, right=False)
    axe2.tick_params(axis="both", which="major", direction="out", width=1.3, length=4, top=False, right=True)
    axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3, top=False, right=False)
    axe.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    axe.yaxis.set_minor_locator(mticker.MultipleLocator(100))
    axe2.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    axe2.yaxis.set_minor_locator(mticker.MultipleLocator(100))





# 初始化图像
fig = plt.figure(figsize=(16, 6), dpi=500)

# 配置 NetCDF 文件路径（注意后缀应为 .nc）
# 修改后的路径定义
im_files = [
    r'Z:/Storage(lustre)\ProjectGroup(lzu_public)\lustre_data\EST_2\N_L_A_trend\Spring_trend/NDVI_spring_trend_Inner_Mongolia.nc', 
    r'Z:/Storage(lustre)\ProjectGroup(lzu_public)\lustre_data\EST_2\N_L_A_trend\Spring_trend/LAI_spring_trend_Inner_Mongolia.nc', 
    r'Z:/Storage(lustre)\ProjectGroup(lzu_public)\lustre_data\EST_2\N_L_A_trend\Spring_trend/Albedo_spring_trend_Inner_Mongolia.nc'
]

m_files = [
    r'Z:/Storage(lustre)\ProjectGroup(lzu_public)\lustre_data\EST_2\N_L_A_trend\Spring_trend/NDVI_spring_trend_Mongolia.nc', 
    r'Z:/Storage(lustre)\ProjectGroup(lzu_public)\lustre_data\EST_2\N_L_A_trend\Spring_trend/LAI_spring_trend_Mongolia.nc', 
    r'Z:/Storage(lustre)\ProjectGroup(lzu_public)\lustre_data\EST_2\N_L_A_trend\Spring_trend/Albedo_spring_trend_with_pvalues_Mongolia.nc'
]

# NetCDF 中的变量名
var_name = ['ndvi_trend', 'lai_trend', 'albedo_trend']
var_name_line = ['ndvi_spring_mean_series', 'lai_spring_mean_series', 'albedo_spring_mean_series']

# Shapefile 路径
shp_im = 'G:/G盘/shp/Inner_Mongolia/Inner_Mongolia.shp'
shp_mn = 'G:/G盘/shp/menggu/menggu.shp'

# 地图子图标题（与变量一致）
titles_map = [
    '(a) NDVI Spring Trend (2005–2023)',
    '(b) LAI Spring Trend (2005–2023)',
    '(c) Albedo Spring Trend (2005–2023)'
]
subplot_index = [231, 232, 233]

# 趋势图设置（左轴min/max，右轴min/max）
trend_titles = [
    '(d) NDVI Interannual Trend (2005–2023)',
    '(e) LAI Interannual Trend (2005–2023)',
    '(f) Albedo Interannual Trend (2005–2023)'
]
ylims = [[0.17, 0.22, 0.12, 0.17], [0.25, 0.45, 0.15, 0.25], [0.18, 0.22, 0.18, 0.24]]
texts = [[0.8, 0.1], [0.8, 0.1], [0.1, 0.8]]
grids = [
    gridspec.GridSpec(20, 144, figure=fig)[11:19, 0:38],
    gridspec.GridSpec(20, 144, figure=fig)[11:19, 50:88],
    gridspec.GridSpec(20, 144, figure=fig)[11:19, 100:138]
]
color_map =[ cmaps.MPL_RdYlGn, cmaps.MPL_RdYlGn, cmaps.MPL_RdYlGn_r]
# 绘制地图子图
for i in range(3):
    plot_map_subplot(m_files[i], im_files[i], var_name[i],color_map[i], subplot_index[i], titles_map[i],
                     fig, shp_im, shp_mn)
    print(f"完成地图子图 {titles_map[i]}")

# 绘制趋势子图
a=[6,3,3]
for i in range(3):
    plot_trend_subplot(grids[i], ylims[i], texts[i], a[i], trend_titles[i],
                       fig, im_files[i], m_files[i], var_name_line[i])
    print(f"完成趋势子图 {trend_titles[i]}")

# 保存图像
plt.subplots_adjust(left=0.06, bottom=0.03, right=0.98, top=0.95, wspace=0.08, hspace=0.22)
plt.savefig('spring_trends_all.png', dpi=500)
print("图像已保存至 spring_trends_all.png")
