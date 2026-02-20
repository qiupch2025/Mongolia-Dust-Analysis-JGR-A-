import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
import numpy as np
import matplotlib.ticker as mticker
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import geopandas as gpd
import matplotlib.colors as colors
import cmaps
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.3

fig = plt.figure(figsize=(16, 16), dpi=500)

gs = gridspec.GridSpec(21, 20, figure=fig)
ax = fig.add_subplot(gs[0:6, 0:20])

dates = pd.date_range(start="2023-03-01", end="2023-05-31", freq="D")

light_colors = [(0.85, 0.34, 0.34),
 (0.85, 0.62, 0.34),
 (0.80, 0.85, 0.34),
 (0.53, 0.85, 0.34),
 (0.34, 0.85, 0.43),
 (0.34, 0.85, 0.71),
 (0.34, 0.71, 0.85),
 (0.34, 0.43, 0.85),
 (0.53, 0.34, 0.85),
 (0.80, 0.34, 0.85),
 (0.85, 0.34, 0.62)]

dark_color = 'darkblue'

all_data = pd.DataFrame()

for date in dates:
    file_path = f'/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/airquality/china_sites_{date.strftime("%Y%m%d")}.csv'
    daily_data = pd.read_csv(file_path, delimiter=',', encoding='utf-8')
    
    hourly_data = daily_data.iloc[3::15].copy()
    
    time_index = pd.date_range(start=date, periods=len(hourly_data), freq='h')
    hourly_data.index = time_index

    all_data = pd.concat([all_data, hourly_data], axis=0)

selected_stations = ["1001A",  "1015A",   "1029A",        "1488A",   "1301A", "1081A",   "1317A",    "1094A",    "1476A" ,"1130A"   ,"1119A"]
selected_labels = ["Beijing", "Tianjin", "Shijiazhuang", "Yinchuan", "Jinan", "Taiyuan", "Zhengzhou","Hohhot","Lanzhou","Harbin","Changchun"]

for station, label, light_color in zip(selected_stations, selected_labels, light_colors[:len(selected_stations)]):
    if station in all_data.columns:
        ax.plot(all_data.index, all_data[station], label=label, color=light_color, alpha=0.5)

    top10 = all_data[station].nlargest(10)

    print(f"{label} top 10 values and indices:")
    for idx, val in top10.items():
        print(f"Index: {idx}, Value: {val}")

average_data = all_data[selected_stations].mean(axis=1, skipna=True)

ax.plot(all_data.index, average_data, color=dark_color, linewidth=3, label='Average')
ax.axhline(y=1000, color='#E3738B', linestyle='--', linewidth=1.5)

ax.set_yscale('linear')
ax.set_ylim(0, 3500)
ax.set_yticks(np.arange(0,2501,200))

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0,3501,500)))

ax.set_title('(a) PM10 Time Series [µg/m$^3$]', loc='left', fontsize=20, pad=8)
ax.legend(fontsize=17, ncol=4, loc = 'upper right', frameon=False)
ax.tick_params(axis='both', labelsize=21)
plt.xticks(rotation=0)

ax.minorticks_on()
ax.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
ax.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
ax.xaxis.set_minor_locator(mticker.MultipleLocator(100000000))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(250))
ax.tick_params(top=False, right=False)
ax.tick_params(axis="x", which="both", top=False)
ax.tick_params(axis="y", which="both", right=False)

axe = plt.subplot(324, projection=ccrs.PlateCarree(), aspect="auto")

sta = pd.read_excel('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/air_quality/stations.xlsx', sheet_name='Station_List_from_20220213')

levels = np.arange(0,160.1,10)

norm = BoundaryNorm(levels, cmaps.WhiteBlueGreenYellowRed.N, clip=True)

pm10_sum = {}
pm10_count = {}
pm10_days = {}

for date in pd.date_range("2023-03-01", "2023-05-31"):
    file_path = f'/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/airquality/china_sites_{date.strftime("%Y%m%d")}.csv'
    try:
        data_daily = pd.read_csv(file_path, delimiter=',', encoding='utf-8')

        for name in data_daily.columns[3::1]:
            if not pd.isna(data_daily[name][0]) and not pd.isna(data_daily[name][3]):
                if name not in pm10_sum:
                    pm10_sum[name] = 0
                    pm10_count[name] = 0
                    pm10_days[name] = 0

                daily_values = data_daily[name].iloc[3::15]
                daily_avg = daily_values.mean()

                pm10_sum[name] += daily_avg
                pm10_count[name] += 1

                if daily_avg > 150:
                    pm10_days[name] += 1

    except FileNotFoundError:
        continue

pm10_avg = {name: pm10_sum[name] / pm10_count[name] if pm10_count[name] > 0 else 0 for name in pm10_sum}

import matplotlib.lines as mlines

def get_marker_size(days):
    if days < 10:
        return 35
    elif 10 <= days < 15:
        return 55
    elif 15 <= days < 20:
        return 75
    elif 20 <= days < 25:
        return 95
    elif 25 <= days < 30:
        return 115
    else:
        return 140

for name in pm10_avg:
    if name in sta['监测点编码'].values:
        lon = float(sta[sta['监测点编码'] == name]['lon'].values[0])
        lat = float(sta[sta['监测点编码'] == name]['lat'].values[0])
        
        pm10 = pm10_avg[name]
        days = pm10_days[name]
        color = cmaps.WhiteBlueGreenYellowRed(norm(pm10))
        size = get_marker_size(days)
        
        axe.scatter(lon, lat, s=size, color=color, transform=ccrs.PlateCarree())

legend_x, legend_y = 93, 50

legend_sizes = [35,55,75,95,115,140]
legend_labels = ["<10",  "10-15", "15-20", "20-25", "25-30", ">30"]

cols = 3
spacing_x = 8
spacing_y = 2

for idx, (size, label) in enumerate(zip(legend_sizes, legend_labels)):
    row = idx // cols
    col = idx % cols

    x_pos = legend_x + col * spacing_x
    y_pos = legend_y - row * spacing_y

    axe.scatter(x_pos, y_pos, s=size, color='gray', edgecolors='black', transform=ccrs.PlateCarree())
    axe.text(x_pos + 1, y_pos, label, fontsize=14, verticalalignment='center', transform=ccrs.PlateCarree())

axe.text(legend_x-1, legend_y + spacing_y, "PM10 > 150 µg/m$^3$ Days", fontsize=16, fontweight='bold', transform=ccrs.PlateCarree())

axe.set_title('(c) Observed PM10 [µg/m$^3$]', loc='left', fontsize=20, pad=8)

axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
axe.set_extent([75, 135, 30, 55], crs=ccrs.PlateCarree())

gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = False, False, False, False
gl.xlocator = mticker.FixedLocator(np.arange(80, 135, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))

axe.set_xticks(np.arange(80, 135, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())

axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())

plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tick_params(top='on', right='on', which='both')

shp_china = shpreader.Reader('/home/qiupch2023/data/shp/china/china.shp').geometries()
axe.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)

gdf = gpd.read_file('/home/qiupch2023/data/shp/china/china.shp')
axe.add_geometries(gdf.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.2)

axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.tick_params(top=False, right=False)
axe.tick_params(axis="x", which="both", top=False)
axe.tick_params(axis="y", which="both", right=False)

cb3 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmaps.WhiteBlueGreenYellowRed), ax=axe, orientation='horizontal', pad=0.1, shrink=1, aspect=22, extend='max')
cb3.outline.set_edgecolor('black')
cb3.ax.tick_params(labelsize=17, width=0, length=0, direction='out', colors='black')
cb3.ax.tick_params(which='both', size=0)
ticks = np.arange(0,161,20)
cb3.set_ticks(ticks)
cb3.set_ticklabels([x for x in ticks])

axe = plt.subplot(323, projection=ccrs.PlateCarree(), aspect="auto")

levels=list(np.arange(0,100000,5000))+list(np.arange(100000,400001,15000))
rgb = [
    [220, 233, 213], [226, 227, 193], [239, 222, 153], [244, 215, 135], [254, 206, 104], [253, 193, 97],
    [251, 178, 86], [246, 139, 61], [244, 119, 53], [241, 95, 46], [239, 82, 39], [233, 65, 36], [219, 45, 38],
    [196, 32, 39], [152, 27, 31]
]
rgb = np.array(rgb) / 255.0

ncfile = nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/MERRA2/MERRA2_DUST/DUST_merra2.nc')
lat = ncfile.variables['lat'][:]
lon = ncfile.variables['lon'][:]
dust = ncfile.variables['dust'][::4,:,:]

dust_mean = np.mean(dust, axis=0)

cmap = cmaps.WhiteBlueGreenYellowRed
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, extend='max')

contourf = plt.contourf(lon, lat, dust_mean*1e9, levels=levels, cmap=cmap, norm=norm, extend='max')

plt.title('(b) Observed Dust [mg/m$^2$]', loc='left', fontsize=20, pad=8)

axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
axe.set_extent([75, 135, 30, 55], crs=ccrs.PlateCarree())

gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = False, False, False, False
gl.xlocator = mticker.FixedLocator(np.arange(80, 135, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))

axe.set_xticks(np.arange(80, 135, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())

axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelsize=21, length=5)

shp_world = shpreader.Reader('/home/qiupch2023/data/shp/world/world.shp').geometries()
shp_china = shpreader.Reader('/home/qiupch2023/data/shp/china/china.shp').geometries()
axe.add_geometries(shp_world, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
axe.add_geometries(shp_china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)

axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.tick_params(top=False, right=False)
axe.tick_params(axis="x", which="both", top=False)
axe.tick_params(axis="y", which="both", right=False)

cb=fig.colorbar(contourf, ax=axe, orientation='horizontal', pad=0.1, shrink=1,aspect=22)
ticks = levels[::4]
cb.set_ticks(ticks)
cb.set_ticklabels([int(x / 1000) for x in ticks])
cb.ax.tick_params(labelsize=17)
cb.ax.tick_params(which='both', size=0)

ncfile=nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/MERRA2/edust_merra2.nc')
ncfile1=nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/ERA5/ERA5_10m.nc')

dust=ncfile.variables['dust_emission'][:,:]
u=np.mean((ncfile1.variables['u10'][:,:,:]),0)
v=np.mean((ncfile1.variables['v10'][:,:,:]),0)
lon1=(ncfile1.variables['longitude'][:])
lat1=(ncfile1.variables['latitude'][:])

axe=plt.subplot(325,projection=ccrs.PlateCarree(),aspect="auto")
lat=(ncfile.variables['lat'][:])
lon=(ncfile.variables['lon'][:])
levels=[0,0.1,0.5,1,2, 3, 4, 5,10,15]
rgb = [
    [255,255,255],
    [77,89,168],
    [15,90,49],
    [27,163,73],
    [168,207,56],
    [253,187,18],
    [246,140,30],
    [240,79,35],
    [237,37,37],
    [237,30,35]
]
rgb = np.array(rgb) / 255.0
new_cmap = ListedColormap(rgb, name='new_cmap')
norm = BoundaryNorm(levels, new_cmap.N, clip=True)
contourf = plt.contourf(lon, lat,dust,levels,colors=rgb, extend='max')
interval=4
u=u[::interval,::interval]
v=v[::interval,::interval]
lat=lat1[::interval]
lon=lon1[::interval]
quiver = plt.quiver(lon, lat,u,v,pivot='tail',width=0.0022, 
                scale=95, color='k', headwidth=2.7,alpha=1,transform=ccrs.PlateCarree())

plt.rcParams['axes.unicode_minus'] = False
axe.set_title('(d) Dust emission [µg/m$^2$/s] and wind at 10m [m/s]', loc='left',fontsize=20, pad=8)
for tick in axe.get_xticklabels() + axe.get_yticklabels():
    tick.set_fontname("Arial")

axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0, color='k')
axe.set_extent([75, 135, 30, 55], crs=ccrs.PlateCarree())

gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = False, False, False, False
gl.xlocator = mticker.FixedLocator(np.arange(80, 135, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))

axe.set_xticks(np.arange(80, 135, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())
axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelcolor='k',length=5)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tick_params(top='on', right='on', which='both') 
shp = shpreader.Reader('/home/qiupch2023/data/shp/world/world.shp').geometries()
axe.add_geometries(shp, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
shp = shpreader.Reader('/home/qiupch2023/data/shp/china/china.shp').geometries()
axe.add_geometries(shp, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))
plt.tick_params(top='on', right='on', which='both') 
rect = patches.Rectangle((129.5, 52),5.5, 3, edgecolor='black', facecolor='w')
axe.add_patch(rect)
axe.quiverkey(quiver,0.95, 0.9, 5, "5m/s",labelpos='N', coordinates='axes', 
              fontproperties={'size':12, 'family':'Arial'})
axe.tick_params(top=False, right=False)
axe.tick_params(axis="x", which="both", top=False)
axe.tick_params(axis="y", which="both", right=False)
cb=fig.colorbar(contourf, ax=axe, orientation='horizontal', pad=0.1, shrink=1,aspect=22)
ticks = levels
cb.set_ticks(ticks)
cb.set_ticklabels(['0','0.1','0.5','1','2','3','4','5','10','15'])
cb.ax.tick_params(labelsize=17)
cb.ax.tick_params(which='both', size=0)

ncfile=nc.Dataset('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/ERA5/ERA5_700hPa.nc')
u=np.mean((ncfile.variables['u'][:,:,:,:]),0)
v=np.mean((ncfile.variables['v'][:,:,:,:]),0)
z=np.mean((ncfile.variables['z'][:,:,:,:]),0)

u=np.squeeze(u)
v=np.squeeze(v)
z=np.squeeze(z)
lon=(ncfile.variables['longitude'][:])
lat=(ncfile.variables['latitude'][:])

axe=plt.subplot(326,projection=ccrs.PlateCarree(),aspect="auto")

levels1=[0]+list(range(2840,3075,13))+[1000000]
levels=[x*10 for x in levels1]
rgb = [
    [255,255,255],
    [206,234,246],
    [154,215,236],
    [113,191,230],
    [76,155,211],
    [39,118,185],
    [34,115,144],
    [28,130,90],
    [34,150,69],
    [92,175,71],
    [168,197,57],
    [247,220,53],
    [245,154,40],
    [240,101,34],
    [234,60,36],
    [215,36,40],
    [196,32,39],
    [169,30,35],
    [139,24,27],
    [112,16,17]
]
rgb = np.array(rgb) / 255.0
contourf = plt.contourf(lon, lat,z,levels,colors=rgb)
interval=5
u=u[::interval,::interval]
v=v[::interval,::interval]
lat=lat[::interval]
lon=lon[::interval]
quiver = plt.quiver(lon, lat,u,v,pivot='tail',width=0.002, 
                scale=220, color='k', headwidth=3,alpha=1,transform=ccrs.PlateCarree())

plt.rcParams['axes.unicode_minus'] = False
axe.set_title('(e) GH [gpm] and wind [m/s] at 700hPa', loc='left',fontsize=20, pad=8)
for tick in axe.get_xticklabels() + axe.get_yticklabels():
    tick.set_fontname("Arial")

axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth=0,color='k')
axe.set_extent([75, 135, 30, 55], crs=ccrs.PlateCarree())

gl = axe.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', linestyle=':')
gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = False, False, False, False
gl.xlocator = mticker.FixedLocator(np.arange(80, 135, 15))
gl.ylocator = mticker.FixedLocator(np.arange(30, 56, 10))

axe.set_xticks(np.arange(80, 135, 15), crs=ccrs.PlateCarree())
axe.set_yticks(np.arange(30, 56, 10), crs=ccrs.PlateCarree())
axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelcolor='k',length=5)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tick_params(top='on', right='on', which='both') 
shp = shpreader.Reader('/home/qiupch2023/data/shp/world/world.shp').geometries()
axe.add_geometries(shp, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
shp = shpreader.Reader('/home/qiupch2023/data/shp/china/china.shp').geometries()
axe.add_geometries(shp, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5, zorder=1)
axe.minorticks_on()
axe.tick_params(axis="both", which="major", direction="out", width=1.3, length=7)
axe.tick_params(axis="both", which="minor", direction="out", width=1.3, length=3.5)
axe.xaxis.set_minor_locator(mticker.MultipleLocator(5))
axe.yaxis.set_minor_locator(mticker.MultipleLocator(5))
plt.tick_params(top='on', right='on', which='both') 
rect = patches.Rectangle((129.5, 52),5.5, 3, edgecolor='black', facecolor='w')
axe.add_patch(rect)
axe.quiverkey(quiver,0.95, 0.9, 10, "10m/s",labelpos='N', coordinates='axes', 
              fontproperties={'size':12, 'family':'Arial'})
axe.tick_params(top=False, right=False)
axe.tick_params(axis="x", which="both", top=False)
axe.tick_params(axis="y", which="both", right=False)
cb=fig.colorbar(contourf,ax=axe,orientation='horizontal', pad=0.1, shrink=1,aspect=22)
ticks = range(28400,30750,260)
cb.set_ticks(ticks)
cb.set_ticklabels(range(2840,3075,26))
cb.ax.tick_params(labelsize=17)
cb.ax.tick_params(which='both', size=0)

plt.subplots_adjust(left=0.06, bottom=0.03, right=0.95, top=0.95, wspace=0.2, hspace=0.1)
plt.savefig('figure2_Arial.png', dpi=500)