import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry import Point
from shapely.prepared import prep
import matplotlib.patches as mpatches
import warnings
import cmaps
warnings.filterwarnings("ignore")

# ==========================================
# 1. 全局样式设置 (出版级标准: Arial + 粗线条)
# ==========================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5        # 边框粗细
plt.rcParams['xtick.major.width'] = 1.5     # X轴刻度线粗细
plt.rcParams['ytick.major.width'] = 1.5     # Y轴刻度线粗细
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['font.weight'] = 'normal'    # 如果需要更粗可以设为 'bold'

# ==========================================
# 2. 配置参数与路径
# ==========================================
input_dir = 'G:/JGRA_DATA/'  # 请确认你的NC文件路径
cities = ["Beijing", "Shenyang", "Lanzhou"] 
file_suffix = "_PM10_CWT_Analysis_Result.nc"

extent = [70, 135, 25, 60]

city_coords = {
    "Beijing":  (116.40, 39.90),
    "Shenyang": (123.43, 41.80),
    "Lanzhou":  (103.83, 36.06)
}

vmin = 0
vmax = 100 

# ==========================================
# 3. 准备 GIS 数据
# ==========================================
print(">>> 正在加载本地 Shapefile...")
try:
    # 加载你指定的本地SHP
    reader_world = shpreader.Reader(r'G:/G盘/shp/world_new/world.shp')
    shp_world = list(reader_world.geometries())
    
    reader_china = shpreader.Reader(r'G:/G盘/shp/china/china.shp')
    shp_china = list(reader_china.geometries())
    print("✅ 本地 SHP 加载成功")
except Exception as e:
    print(f"❌ SHP 加载失败 (请检查路径): {e}")
    shp_world = []
    shp_china = []

# 准备掩码 (用于计算贡献)
print(">>> 准备计算掩码...")
reader = shpreader.Reader(shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries'))
countries = list(reader.records())
mongolia_geom, china_geom = None, None
for c in countries:
    if c.attributes['NAME'] == 'Mongolia': mongolia_geom = c.geometry
    elif c.attributes['NAME'] == 'China': china_geom = c.geometry

if not mongolia_geom or not china_geom:
    print("❌ NaturalEarth 数据缺失，无法计算柱状图")
    exit()

mongolia_prep = prep(mongolia_geom)
china_prep = prep(china_geom)

def calculate_contribution(ds):
    lats, lons = ds.lat.values, ds.lon.values
    cwt = ds['CWT_Final'].values
    cwt = np.nan_to_num(cwt)
    sum_m, sum_c, sum_total = 0, 0, np.sum(cwt)
    rows, cols = np.where(cwt > 0)
    for r, c in zip(rows, cols):
        p = Point(lons[c], lats[r])
        val = cwt[r, c]
        if mongolia_prep.contains(p): sum_m += val
        elif china_prep.contains(p): sum_c += val
    sum_o = sum_total - sum_m - sum_c
    return sum_m, sum_c, max(0, sum_o), sum_total

# ==========================================
# 4. 绘图主程序
# ==========================================
def main():
    print(">>> 开始绘制最终图...")
    
    # 画布设置 (13x11英寸)
    fig = plt.figure(figsize=(13, 9), dpi=300)
    
    # 颜色设置
    cmap = cmaps.WhiteBlueGreenYellowRed
    cmap.set_bad(alpha=0) # NaN透明
    
    stats_data = {"Mongolia": [], "China": [], "Others": []}
    mesh = None 
    
    # --- 循环绘制 3 个地图 ---
    for i, city in enumerate(cities):
        idx = i + 1 
        ax = fig.add_subplot(2, 2, idx, projection=ccrs.PlateCarree(), aspect="auto")

        
        fpath = os.path.join(input_dir, f"{city}{file_suffix}")
        if not os.path.exists(fpath): continue
        ds = xr.open_dataset(fpath)
        
        # 计算
        m, c, o, t = calculate_contribution(ds)
        stats_data["Mongolia"].append(m/t*100)
        stats_data["China"].append(c/t*100)
        stats_data["Others"].append(o/t*100)
        
        # ---------------------------
        # A. 地理底图填充 (灰地蓝海)
        # ---------------------------
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # 陆地填充灰色
        ax.add_feature(cfeature.LAND, facecolor='#F0F0F0', zorder=0)
        # 海洋填充蓝色
        ax.add_feature(cfeature.OCEAN, facecolor='#B0C4DE', zorder=0)
        # 湖泊
        ax.add_feature(cfeature.LAKES, facecolor='#B0C4DE', edgecolor='gray', linewidth=0.5, zorder=0)

        # ---------------------------
        # B. 绘制潜在源区 (黄褐色填充)
        # ---------------------------
        # 使用 contourf 填充 mask=1 的区域
        # levels=[0.5, 1.5] 确保只选取值为1的区域
        # 颜色 #C2B280 (Ecru/Khaki/黄褐色), alpha=0.5 半透明
        ax.contourf(
            ds.lon, ds.lat, ds['Dust_Source_Mask'], 
            levels=[0.5, 1.5], 
            colors=['#C2B280'], 
            alpha=0.5, 
            transform=ccrs.PlateCarree(), 
            zorder=1
        )
        
        # ---------------------------
        # C. 绘制 CWT 热力图
        # ---------------------------
        plot_cwt = ds['CWT_Final'].where(ds['CWT_Final'] > 0)
        mesh = plot_cwt.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax,
            add_colorbar=False, rasterized=True, zorder=2
        )
        
        # ---------------------------
        # D. 叠加本地 SHP (World & China)
        # ---------------------------
        # 世界地图 (灰色背景线)
        if shp_world:
            ax.add_geometries(shp_world, ccrs.PlateCarree(),
                              edgecolor='gray', facecolor='none', 
                              linewidth=1.0, zorder=3)
        # 中国地图 (黑色加粗强调)
        if shp_china:
            ax.add_geometries(shp_china, ccrs.PlateCarree(),
                              edgecolor='gray', facecolor='none', 
                              linewidth=1.0, zorder=4)

        # ---------------------------
        # E. 装饰要素
        # ---------------------------
        # 城市打点
        cx, cy = city_coords[city]
        ax.plot(cx, cy, marker='*', color='r', ms=22, mec='k', mew=1.5, transform=ccrs.PlateCarree(), zorder=10)
        
        # 坐标轴
        ax.set_xticks(np.arange(70, 140, 15), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(30, 65, 10), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        # 标题
        letter = chr(97 + i)
        ax.set_title(f"({letter}) {city}", loc='left', fontweight='bold', pad=2, fontsize=20)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # --- 色标 (CAX 指定位置) ---
    # [left, bottom, width, height]
    # 放在两个地图的下方
    cbar_ax = fig.add_axes([0.08, 0.08, 0.4, 0.02]) 
    # 1. 创建色标
    cb = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal', extend='max')

    # 2. 设置刻度字体大小为 16
    cb.ax.tick_params(labelsize=16)
    cb.set_label('Weighted CWT Concentration ($\mu g/m^3$)', fontweight='bold',fontsize=16)
    cb.outline.set_linewidth(2) # 色标框也加粗

    # ==========================================
    # 4. 右下角：竖向堆叠柱状图
    # ==========================================
    ax_bar = fig.add_subplot(2, 2, 4)
    
    # 配色
    c_m, c_c, c_o = "#D14949", "#E39755", '#F0F0F0'
    
    x_pos = np.arange(len(cities))
    bar_width = 0.55
    
    # 绘制柱子 (带黑色边框 linewidth=2)
    p1 = ax_bar.bar(x_pos, stats_data["Mongolia"], width=bar_width, color=c_m, label='Mongolia', 
                   edgecolor='black', linewidth=2, zorder=3)
    p2 = ax_bar.bar(x_pos, stats_data["China"], width=bar_width, bottom=stats_data["Mongolia"], 
                   color=c_c, label='China', edgecolor='black', linewidth=2, zorder=3)
    
    bottom_others = [m+c for m,c in zip(stats_data["Mongolia"], stats_data["China"])]
    p3 = ax_bar.bar(x_pos, stats_data["Others"], width=bar_width, bottom=bottom_others, 
                   color=c_o, edgecolor='black', linewidth=2, label='Others', zorder=3)
    
    # 添加百分比标签
    def add_labels(stats, bottom_vals, color='white'):
        for i, val in enumerate(stats):
            if val > 5:
                height = bottom_vals[i] + val/2 if bottom_vals else val/2
                ax_bar.text(i, height, f"{val:.1f}%", ha='center', va='center', 
                           color=color, fontweight='bold', fontsize=18)
    add_labels(stats_data["Mongolia"], None, 'white')
    add_labels(stats_data["China"], stats_data["Mongolia"], 'white')

    # 柱状图修饰
    ax_bar.set_ylim(0, 100)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(cities, fontweight='bold', fontsize=16)
    ax_bar.tick_params(axis='x', labelsize=16)
    ax_bar.tick_params(axis='y', labelsize=16)
    ax_bar.set_ylabel('Contribution Percentage (%)', fontweight='bold', fontsize=16)
    ax_bar.set_title("(d) Source Contribution", loc='left', fontweight='bold', pad=2, fontsize=20)
    
    # 边框设置
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    # 剩下的左和下边框加粗
    ax_bar.spines['left'].set_linewidth(1.5)
    ax_bar.spines['bottom'].set_linewidth(1.5)
    
    ax_bar.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.1), 
        ncol=3, 
        frameon=False, 
        fontsize=20,
        handletextpad=0.1,  # 【修改1】图块与文字的距离 (单位是字体大小的倍数，数值越小越近)
        columnspacing=0.6   # 【修改2】两个图例之间的距离 (数值越大越宽)
    )

    # 布局调整 (Auto Fill)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15, wspace=0.15, hspace=0.25)

    save_path = os.path.join(input_dir, "Figure3_Final_Pub_v2.png")
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight') # PDF矢量图
    plt.savefig('./Figure3_Final_Pub_v2.png', bbox_inches='tight', dpi=600)
    print(f"✅ 图表已保存: {save_path}")
    # plt.show()

if __name__ == "__main__":
    main()