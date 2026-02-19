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

warnings.filterwarnings("ignore")

# ==========================================
# 1. Parameter and Path Configuration
# ==========================================
# Please replace these with your actual local paths
base_hysplit_path = 'G:/G_Drive/Mongolian_Desertification_Dust_Project/Dust_Work/EST_2/HYSPLIT_DATA/aa_HYSPLIT_NEW/'
pm10_dir = 'G:/G_Drive/Mongolian_Desertification_Dust_Project/aa_revision1/airquality/Stations_20230101-20231007/'
output_dir = 'G:/JGRA_DATA/' 
ndvi_path = r"Z:\Storage(lustre)\ProjectGroup(lzu_public)\lustre_data\EST_2\aaa_new\ndvi\ndvi_monthly_avg_2003-2023.nc"

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

city_configs = {
    "Beijing":  "1001A",
    "Lanzhou":  "1476A",
    "Shenyang": "1099A",
    "Taiyuan":  "1081A"
}

# Set grid resolution (0.5 degree)
res = 0.5
lon_range = [70, 135]
lat_range = [25, 60]

lons_arr = np.arange(lon_range[0], lon_range[1] + res, res, dtype=float)
lats_arr = np.arange(lat_range[0], lat_range[1] + res, res, dtype=float)

# ==========================================
# 2. Core: NDVI Data Loading and Reconstruction
# ==========================================
def load_clean_ndvi(path):
    print(f">>> Reading NDVI file: {os.path.basename(path)}")
    ds = xr.open_dataset(path)
    
    # 1. Identify coordinate variable names
    lat_arr = None
    lon_arr = None
    for key in ds.variables:
        if 'lat' in key.lower(): lat_arr = ds[key].values
        if 'lon' in key.lower(): lon_arr = ds[key].values
            
    if lat_arr is None or lon_arr is None:
        raise ValueError("❌ Could not find latitude/longitude variables in the file!")

    # 2. Extract 2023 Spring data (March, April, May)
    # Assuming dimensions are (time, lon, lat) or (time, lat, lon)
    # Assuming the last year in the dataset is 2023
    raw_ndvi = ds['ndvi']
    all_vals = raw_ndvi.values 
    
    print("   -> Extracting 2023 Spring (MAM) average...")
    
    # Logic for 4D arrays [month, year, lat, lon]
    if all_vals.ndim == 4:
        # Indices 2, 3, 4 correspond to March, April, May; index -1 is the last year
        spring_vals = all_vals[2:5, -1, :, :] 
        spring_avg = np.nanmean(spring_vals, axis=0)
    else:
        # Fallback for other dimension structures
        spring_avg = np.nanmean(all_vals, axis=tuple(range(all_vals.ndim - 2)))

    # 3. Transpose check (ensure shape is Lat x Lon)
    target_shape = (len(lat_arr), len(lon_arr))
    if spring_avg.shape != target_shape:
        print("   -> Transposition detected, correcting dimensions...")
        spring_avg = spring_avg.T

    # 4. Construct DataArray
    clean_da = xr.DataArray(
        spring_avg,
        coords={'lat': lat_arr, 'lon': lon_arr},
        dims=('lat', 'lon')
    )
    return clean_da.sortby('lat').sortby('lon')

try:
    spring_avg_raw = load_clean_ndvi(ndvi_path)
    print("✅ NDVI data prepared successfully.")
except Exception as e:
    print(f"❌ Failed to read NDVI: {e}")
    exit()

# ==========================================
# 3. Mask Generation (Water Removal + Threshold + Lakes)
# ==========================================
n_rows = int(lats_arr.shape[0])
n_cols = int(lons_arr.shape[0])
print(f"   -> Target grid size: {n_rows} x {n_cols}")

ndvi_weight_mask = np.zeros((n_rows, n_cols))

# Extract data for performance
data_lats = spring_avg_raw.lat.values
data_lons = spring_avg_raw.lon.values
data_vals = spring_avg_raw.values

print("   -> Step A: Calculating grid attributes (pixel-level water removal)...")

for i in range(n_rows):
    for j in range(n_cols):
        lat_s = float(lats_arr[i])
        lon_s = float(lons_arr[j])
        
        # 1. Extract all NDVI pixels within the current grid
        mask_lat = (data_lats >= lat_s) & (data_lats < lat_s + res)
        mask_lon = (data_lons >= lon_s) & (data_lons < lon_s + res)
        
        valid_rows = data_vals[mask_lat, :]
        valid_pixels = valid_rows[:, mask_lon]
        
        # 2. [Core] Pixel-level mask: Keep only land pixels (NDVI >= 0)
        valid_land_pixels = valid_pixels[valid_pixels >= 0]
        
        # 3. Calculate land average and determine source eligibility
        if valid_land_pixels.size > 0:
            grid_spatial_mean = np.nanmean(valid_land_pixels)
            if not np.isnan(grid_spatial_mean):
                # Threshold for sparse vegetation (potential dust source)
                if grid_spatial_mean < 0.12: 
                    ndvi_weight_mask[i, j] = 1.0
                else:
                    ndvi_weight_mask[i, j] = 0.0
        else:
            # Grid consists entirely of water
            ndvi_weight_mask[i, j] = 0.0

print("   -> Step B: Applying GIS Lake Mask (removing Lake Baikal, etc.)...")

# Load Natural Earth lake data
reader = shpreader.Reader(shpreader.natural_earth(resolution='50m', category='physical', name='lakes'))
all_lakes = list(reader.geometries())

# Spatial filtering for performance (keep only lakes within study area)
study_area = box(lon_range[0]-2, lat_range[0]-2, lon_range[1]+2, lat_range[1]+2)
relevant_lakes = [lake for lake in all_lakes if lake.intersects(study_area)]
lake_preps = [prep(lake) for lake in relevant_lakes]

# Check if points marked as sources fall within lakes
rows_idx, cols_idx = np.where(ndvi_weight_mask == 1)
removed_count = 0

for r, c in zip(rows_idx, cols_idx):
    # Determine grid center coordinates
    lat_p = lats_arr[r] + res/2 
    lon_p = lons_arr[c] + res/2
    p = Point(lon_p, lat_p)
    
    for lake in lake_preps:
        if lake.contains(p):
            ndvi_weight_mask[r, c] = 0.0 # Force set to non-source
            removed_count += 1
            break 

print(f"   -> Masking complete. Grids removed via lake mask: {removed_count}")

# Verify mask effect
plt.figure(figsize=(8, 5))
plt.imshow(ndvi_weight_mask, origin='lower', 
           extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]], 
           cmap='Reds')
plt.colorbar(label='Is Source? (1=Yes)')
plt.title("Final Dust Source Mask")
plt.show()

# ==========================================
# 4. Prepare PM10 Data
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
        # Ensure timezone is CST (Asia/Shanghai) to match HYSPLIT local time
        all_data.index = all_data.index.tz_localize("Asia/Shanghai") if all_data.index.tz is None else all_data.index
    return all_data

print(">>> Loading PM10 data...")
pm10_master = get_pm10_lookup_table(pm10_dir)

# ==========================================
# 5. CWT Calculation (with Weights)
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
    print(f"\n--- Processing City: {city} ---")
    traj_dir = os.path.join(base_hysplit_path, city) + '/'
    clus_file = os.path.join(traj_dir, "julei/CLUSLIST_4")
    out_nc = os.path.join(output_dir, f"{city}_PM10_CWT_Analysis_Result.nc")
    
    if not os.path.exists(clus_file): 
        print(f"⚠️ Clustering file not found: {clus_file}")
        continue

    sum_conc = np.zeros((n_rows, n_cols))
    sum_count = np.zeros((n_rows, n_cols))

    try:
        clus_df = pd.read_csv(clus_file, sep=r'\s+', header=None, engine='python',
                              names=["C", "N", "Y", "M", "D", "H", "I", "Path"])
        
        success_traj = 0
        for _, row in clus_df.iterrows():
            fpath = os.path.join(traj_dir, os.path.basename(str(row['Path']).strip("'")))
            
            # Time conversion: UTC -> Local
            dt_utc = datetime(int(row['Y'])+2000, int(row['M']), int(row['D']), int(row['H']))
            dt_loc = pytz.utc.localize(dt_utc).astimezone(pytz.timezone('Asia/Shanghai'))
            
            # PM10 matching
            if dt_loc not in pm10_master.index: continue
            val = pm10_master.loc[dt_loc, sid]
            if pd.isna(val): continue
            
            # Trajectory parsing
            points = parse_tdump(fpath)
            if points is None: continue
            
            success_traj += 1
            
            # Grid accumulation
            for lat, lon in points:
                if (lat_range[0] <= lat <= lat_range[1]) and (lon_range[0] <= lon <= lon_range[1]):
                    r_idx = int((lat - lat_range[0]) // res)
                    c_idx = int((lon - lon_range[0]) // res)
                    
                    if 0 <= r_idx < n_rows and 0 <= c_idx < n_cols:
                        sum_conc[r_idx, c_idx] += float(val)
                        sum_count[r_idx, c_idx] += 1
        
        if success_traj > 0:
            # 1. Base CWT
            cwt_base = np.divide(sum_conc, sum_count, out=np.zeros_like(sum_conc), where=sum_count!=0)
            
            # 2. Weighting function (Polissar et al.)
            # Based on average trajectory endpoints per non-zero grid
            v_counts = sum_count[sum_count > 0]
            avg = np.mean(v_counts) if len(v_counts) > 0 else 1
            
            w = np.ones_like(sum_count)
            w[sum_count <= 3*avg] = 0.70
            w[sum_count <= 2*avg] = 0.42
            w[sum_count <= avg] = 0.17
            
            cwt_weighted = cwt_base * w
            
            # 3. Apply NDVI physical constraints
            cwt_final = cwt_weighted * ndvi_weight_mask 
            
            # 4. Save results to NetCDF
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
            print(f"✅ Calculation for {city} finished. Saved to {out_nc}")
        else:
            print(f"⚠️ Valid trajectory count for {city} is 0")
            
    except Exception as e:
        print(f"⚠️ Error processing {city}: {e}")

print("\n>>> All tasks completed.")
