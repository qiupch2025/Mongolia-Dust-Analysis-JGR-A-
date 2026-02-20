import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import geopandas as gpd
from scipy.optimize import nnls
from sklearn.metrics import r2_score
from matplotlib.ticker import MultipleLocator

# ================== 1. 基础配置与路径 ==================
data_file = "G:/JGRA_DATA/merra2_data3/processed_data/merra2_dust_ts_1980_2025.csv"
clim_file = "G:/JGRA_DATA/merra2_data3/processed_data/climatology_emiss.nc"

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
FRAME_LW = 3  # 严格保持画框粗细

# --- 颜色分级与色板 ---
levels = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 15]
rgb = np.array([
    [255, 255, 255], [77, 89, 168], [15, 90, 49], [27, 163, 73], [168, 207, 56],
    [253, 187, 18], [246, 140, 30], [240, 79, 35], [237, 37, 37], [237, 30, 35], [150, 20, 20]
]) / 255.0
custom_cmap = ListedColormap(rgb)

receptors = ["NEC", "NC", "NWC"]
sources = ["Mongolia", "Xinjiang", "Inner_Mongolia"]
source_shps = [
    {"path": "G:/G盘/shp/menggu/menggu.shp", "color": "k", "lw": 4, "label": "Mongolia"},
    {"path": "G:/JGRA_DATA/shp/xinjiang.shp", "color": "k", "lw": 4, "label": "Xinjiang"},
    {"path": "G:/G盘/shp/Inner_Mongolia/Inner_Mongolia.shp", "color": "k", "lw": 4, "label": "Inner Mongolia"},
]

# ================== 2. 数据加载 ==================
df = pd.read_csv(data_file, index_col='time', parse_dates=True)
df_annual = df.groupby(df.index.year).mean()
ds_clim = xr.open_dataset(clim_file)
da_plot = ds_clim['dust_emiss']

# ================== 3. 开始绘图 (严格维持 3x3 布局) ==================
fig = plt.figure(figsize=(24, 28), dpi=300)
gs = fig.add_gridspec(3, 4, hspace=0.2, wspace=0.8, height_ratios=[0.4, 0.55, 0.6], width_ratios=[0, 1.3, 1, 0.1])

# ------------------ Row 1, Col 0: (a) 时间序列图 (三区域对比) ------------------
# ------------------ Row 1, Col 0: (a) 时间序列图 (精简版) ------------------
ax_ts = fig.add_subplot(gs[0, 1])

# 1. 绘制受体区浓度 (统一在主轴)
colors_rec = ['#27ae60', '#2980b9', '#8e44ad']
lns_rec = []

for idx, rec in enumerate(receptors):
    # 直接绘制浓度序列
    rec_data = df_annual[f'rec_{rec}']
    ln = ax_ts.plot(df_annual.index, rec_data, color=colors_rec[idx], 
                    linewidth=4, label=f'{rec}') # 适当加粗
    lns_rec.extend(ln)

# 2. 设置轴标签与范围
ax_ts.set_ylabel('Dust Conc. [$mg/m^2$]', fontsize=30, fontweight='bold', color='#2c3e50')
ax_ts.set_ylim([70, 420]) # 保持与之前一致的范围

# 3. 标注 2023 研究年份 (可选，若不需要可删掉)
if 2023 in df_annual.index:
    ax_ts.axvline(x=2023, color='red', linestyle='--', alpha=0.5, linewidth=3)
    ax_ts.text(2023.5, 390, '2023', color='red', fontsize=28, fontweight='bold')

# 4. 图例细节
ax_ts.legend(handles=lns_rec, loc='upper left', fontsize=24, frameon=False, ncol = 2)
ax_ts.set_title("(a) Multi-Region Time Series", loc='left', fontsize=30, fontweight='bold', pad=20)

# 5. 副刻度与轴线精修 (严格保持 FRAME_LW)
ax_ts.minorticks_on()
ax_ts.xaxis.set_minor_locator(MultipleLocator(1))
ax_ts.tick_params(labelsize=28, width=FRAME_LW, length=10)
# 去掉右侧刻度线，保持简洁
ax_ts.tick_params(axis='y', which='both', right=False)

for spine in ax_ts.spines.values(): 
    spine.set_linewidth(FRAME_LW)

# ------------------ Row 1, Col 1: (b) 空间填色图 ------------------
ax_map = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree(), aspect="auto")
ax_map.set_aspect('auto', adjustable='datalim')
ax_map.set_extent([73, 130, 32, 53], crs=ccrs.PlateCarree())
cf = ax_map.contourf(da_plot.lon, da_plot.lat, da_plot, levels=levels, colors=rgb, extend='max', transform=ccrs.PlateCarree())
ax_map.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=2)
for shp in source_shps:
    gdf = gpd.read_file(shp["path"])
    gdf.boundary.plot(ax=ax_map, color=shp["color"], linewidth=shp["lw"], transform=ccrs.PlateCarree())
from shapely.ops import nearest_points
from shapely.geometry import Point

# 1. 手动指定文字（经纬度）位置 
# 注意：Key 必须与 source_shps 中的 label 完全一致
label_positions = {
    "Mongolia": (105, 55),       # 北部边缘 (略高于 53)
    "Xinjiang": (85, 30),       # 西南部边缘 (略低于 32)
    "Inner Mongolia": (110, 33)  # 南部边缘 (注意这里是空格，匹配 shp 里的 label)
}

for shp in source_shps:
    # A. 读取并绘制边界
    gdf = gpd.read_file(shp["path"])
    geom = gdf.geometry.unary_union
    gdf.boundary.plot(ax=ax_map, color=shp["color"], linewidth=shp["lw"], transform=ccrs.PlateCarree(), zorder=5)
    
    label_name = shp["label"]
    # 自动兼容下划线或空格的匹配逻辑
    pos = label_positions.get(label_name) or label_positions.get(label_name.replace(" ", "_"))
    
    if pos is None:
        continue
        
    txt_lon, txt_lat = pos
    
    # B. 计算文字点到 SHP 边界最近的点
    txt_point = Point(txt_lon, txt_lat)
    # 确保从 boundary 中找点，这样线会停在地图边缘
    _, nearest_geom_point = nearest_points(txt_point, geom.boundary)
    target_lon, target_lat = nearest_geom_point.x, nearest_geom_point.y

    # C. 绘制标注（无边框版本）
    ax_map.annotate(
        label_name.replace("_", " "),     # 显示时统一美化
        xy=(target_lon, target_lat),      # 连线终点：区域边缘
        xytext=(txt_lon, txt_lat),        # 连线起点：文字位置
        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax_map),
        textcoords=ccrs.PlateCarree()._as_mpl_transform(ax_map),
        fontsize=28,                      # 调大字号匹配大图
        fontweight='bold',
        color='black',
        # 移除边框：通过设置 alpha=0 或直接删除 bbox
        bbox=dict(facecolor='none', edgecolor='none', alpha=0), 
        # 连线配置
        arrowprops=dict(
            arrowstyle='-',               # 纯直线
            connectionstyle="arc3,rad=0", 
            color='black', 
            lw=3,                         # 线条加粗
            shrinkA=8,                    # 离文字稍微远一点，避免粘连
            shrinkB=0                     # 紧贴地图边界
        ),
        ha='center',
        va='center',
        zorder=10
    )
# 地图刻度与副刻度精修
ax_map.set_xticks(np.arange(80, 135, 15), crs=ccrs.PlateCarree())
ax_map.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())
ax_map.xaxis.set_major_formatter(LongitudeFormatter())
ax_map.yaxis.set_major_formatter(LatitudeFormatter())
ax_map.tick_params(axis='both', which='major', labelsize=28, direction='out', length=10, width=FRAME_LW)
ax_map.tick_params(axis='both', which='minor', direction='out', length=5, width=1.5)
ax_map.minorticks_on()
ax_map.xaxis.set_minor_locator(mticker.MultipleLocator(2))
ax_map.yaxis.set_minor_locator(mticker.MultipleLocator(1))
for spine in ax_map.spines.values(): spine.set_linewidth(FRAME_LW)

# Colorbar (维持侧边位置)
cbar_ax = fig.add_axes([0.79, 0.705, 0.012, 0.175]) 
cb = fig.colorbar(cf, cax=cbar_ax)
cb.ax.tick_params(labelsize=28)
cb.set_label('$\mu g/m^2/s$', fontsize=28)
cb.outline.set_linewidth(FRAME_LW)
ax_map.set_title("(b) Climatology Distribution", loc='left', fontsize=30, fontweight='bold', pad=20)

# ------------------ Row 2 & 3: 归因分析与验证 (严格副刻度) ------------------
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.25, height_ratios=[0.4, 0.5, 0.6], width_ratios=[1, 1, 1])
bar_colors = np.array([[77, 89, 168], [253, 187, 18], [237, 37, 37]]) / 255.0

for i, rec in enumerate(receptors):
    y = df[f"rec_{rec}"]
    X = df[[f"src_{s}" for s in sources]]
    A = np.c_[X.values, np.ones(len(X))]
    coefs, _ = nnls(A, y.values)
    betas = coefs[:len(sources)]
    y_pred = A @ coefs
    
    # --- Row 2: 贡献占比柱状图 ---
    ax_bar = fig.add_subplot(gs[1, i])
    contrib_abs = X.multiply(betas, axis=1)
    contrib_pct = contrib_abs.groupby(contrib_abs.index.year).mean()
    contrib_pct = contrib_pct.div(contrib_pct.sum(axis=1), axis=0) * 100
    contrib_pct.plot(kind='bar', stacked=True, ax=ax_bar, color=bar_colors, width=0.8, legend=False)
    
    ax_bar.set_title(f"({chr(99+i)}) {rec} Source", fontsize=30, fontweight='bold', loc='left', pad=20)
    if i == 0: ax_bar.set_ylabel("Contribution Share (%)", fontsize=30)
    ax_bar.set_xlabel("Contribution Share (%)", fontsize=0)

    ax_bar.set_xticks(np.arange(0, len(contrib_pct), 10))
    ax_bar.set_xticklabels(contrib_pct.index[::10], rotation=0, fontsize=25)
    # 副刻度
    ax_bar.minorticks_on()
    ax_bar.xaxis.set_minor_locator(MultipleLocator(5))
    ax_bar.yaxis.set_minor_locator(MultipleLocator(5))
    ax_bar.tick_params(axis='both', which='major', length=10, width=FRAME_LW, labelsize=28)
    ax_bar.tick_params(axis='both', which='minor', length=5, width=FRAME_LW*0.7)
    ax_bar.set_ylim(0, 100)
    for spine in ax_bar.spines.values(): spine.set_linewidth(FRAME_LW)
    if i == 2: ax_bar.legend(sources, loc='upper center', bbox_to_anchor=(-0.6, -0.1), ncol=3, fontsize=32, frameon=False)

# --- Row 3: 散点图验证 (手动精准布局红星与文字) ---
    ax_scat = fig.add_subplot(gs[2, i])
    
    # 1. 统计指标计算
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(((y - y_pred) ** 2).mean())
    nrmse = (rmse / y.mean()) * 100
    
    # 计算 2023 年残差与 Z-score
    residuals = y - y_pred
    res_2023 = residuals[y.index.year == 2023].mean()
    z_score_2023 = (res_2023 - residuals.mean()) / residuals.std()
    
    # 2. 绘制背景散点与 1:1 线
    ax_scat.scatter(y, y_pred, alpha=0.5, c='gray', s=200, edgecolors='none', zorder=2)
    max_val = max(y.max(), y_pred.max()) * 1.1
    ax_scat.plot([0, max_val], [0, max_val], 'r--', lw=3, zorder=3)
    
    # 3. 绘制图中 2023 年样本点 (红星)
    y_2023 = y[y.index.year == 2023].mean()
    yp_2023 = y_pred[y.index.year == 2023].mean()
    ax_scat.scatter(y_2023, yp_2023, color='red', marker='*', s=2500, 
                    edgecolors='black', linewidths=2, zorder=5)
    
    # 4. 左上角：模型指标文本框
    stats_left = (f"$R^2 = {r2:.2f}$\n"
                  f"RMSE = {rmse:.1f}\n"
                  f"NRMSE = {nrmse:.1f}%")
    ax_scat.text(0.05, 0.95, stats_left, transform=ax_scat.transAxes, 
                 fontsize=32, va='top', ha='left',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # 5. 右下角自定义“图例”：文字右对齐 + 独立红星
    # 5.1 绘制文字说明 (右对齐 ha='right')
    # \n 前后控制行间距
    # 计算原始残差的物理值 (观测 - 模拟)
    res_raw_2023 = y_2023 - yp_2023 
    
    # 修改右下角文本内容
    # 增加一行 Residual，保留一位小数
    label_text = (f"Spring 2023\n"
                  f"Res = {res_raw_2023:.1f} $mg/m^2$\n"
                  f"Z-score = {z_score_2023:.3f}")
    
    ax_scat.text(0.95, 0.05, label_text, transform=ax_scat.transAxes, 
                 fontsize=32, fontweight='bold', va='bottom', ha='right', 
                 color='red', linespacing=1.3)
    
    # 5.2 绘制独立的红星 (位置通过 transAxes 锁定)
    # 这里的 xy 位置 (0.68, 0.12) 需要根据您的文字宽度微调，使其刚好在 "Year 2023" 左侧
    ax_scat.scatter(0.42, 0.255, color='red', marker='*', s=1200, 
                    edgecolors='black', linewidths=1.5,
                    transform=ax_scat.transAxes, zorder=6)

    # 6. 坐标轴与画框精修
    ax_scat.set_title(f"({chr(102+i)}) {rec} Validation", fontsize=30, fontweight='bold', loc='left', pad=20)
    ax_scat.set_xlabel("Observed [$mg/m^2$]", fontsize=30)
    if i == 0: ax_scat.set_ylabel("Predicted [$mg/m^2$]", fontsize=30)
    
    ax_scat.set_xlim(0, max_val)
    ax_scat.set_ylim(0, max_val)
    ax_scat.tick_params(labelsize=28, width=FRAME_LW, length=10)
    ax_scat.minorticks_on()
    for spine in ax_scat.spines.values(): 
        spine.set_linewidth(FRAME_LW)

plt.savefig('./Dust_Analysis_3x3_Final_Restored.png', bbox_inches='tight', dpi=600)
# plt.show()