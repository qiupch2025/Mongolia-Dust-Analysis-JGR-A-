% clc;
% clear;

NDVIS = 0.05;
NDVIV = 0.90;

% Read NDVI data for 2005–2023
disp('Starting to read data...')
ndviData = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'ndvi_MOD13C2');
lon_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'longitude');
lat_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'latitude');

% Time dimension
years = 2005:2023;
num_years = length(years);
[lats, lons, ~, num_months] = size(ndviData);

% Initialize storage for detrended NDVI data
ndvi_detrended = nan(lats, lons, num_months);

disp('Removing long-term NDVI trends for 2005–2023 while preserving seasonal variability')

% Loop over each grid cell
for r = 1:lats
    disp(['Processing row: ', num2str(r)]) % Show progress
    for c = 1:lons
        for month = 1:num_months
            % Extract the NDVI time series (2005–2023) for the current grid cell and month
            ndviSeries = squeeze(ndviData(r, c, :, month));

            % Ensure data are valid
            if all(~isnan(ndviSeries))
                % Least-squares linear trend fit: y = a*t + b
                p_ndvi = polyfit(years, ndviSeries, 1);

                % Compute the trend component at 2023
                trend_ndvi_at_2023 = p_ndvi(1) * (2023 - 2005);

                % Detrend
                ndvi_detrended(r, c, month) = ndviData(r, c, end, month) - trend_ndvi_at_2023;
            end
        end
    end
end

disp('Detrended NDVI computation completed')

% %% **Apply global NDVI growth rate**
% ndvi_growth_rate = [0.01, 0.008, 0.006]; % Growth rates for March, April, May
%
% for month_idx = 1:3  % March, April, May
%     month = month_idx + 2; % Corresponding to March, April, May
%     % Compute globally increased NDVI
%     ndvi_detrended(:,:,month) = ndvi_detrended(:,:,month) * (1 + ndvi_growth_rate(month_idx) * num_years);
% end
%
% disp('Global NDVI growth rate applied')

%% **Compute GVF (Green Vegetation Fraction)**
GVF = (ndvi_detrended - NDVIS) ./ (NDVIV - NDVIS);

% Constrain GVF to [0, 1]
GVF(GVF < 0) = 0;
GVF(GVF > 1) = 1;

disp('Detrended GVF computation completed')

%% **Read target grid**
geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2005/geo_em.d01_2005.nc';
targetLonGrid = ncread(geoFilePath,'CLONG');
targetLatGrid = ncread(geoFilePath,'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

% Read polygon data
s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

% Determine which points are inside the polygon
inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);

disp('Computing mean detrended GVF on the target grid, restricted to the inPoly region')

% Initialize GVF for the 119x99 grid
GVF_grid = nan(nx-1, ny-1, num_months);

% Loop over the target grid
for i = 1:nx-1
    i
    for j = 1:ny-1
        if inPoly(i, j)
            % Get boundaries of the current 119x99 grid cell
            lat_min = min([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lat_max = max([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lon_min = min([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);
            lon_max = max([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);

            % Extract high-resolution data within the current grid cell
            mask = (lat_highRes >= lat_min) & (lat_highRes <= lat_max) & ...
                   (lon_highRes >= lon_min) & (lon_highRes <= lon_max);

            for month = 1:num_months
                sub_data = GVF(:,:,month);
                sub_data = sub_data(mask);
                
                % Compute mean GVF
                GVF_grid(i,j,month) = nanmean(sub_data, 'all');
            end
        end
    end
end

disp('Mean detrended GVF computed; writing to geo file')

%% **Replace GVF data in the geo file, affecting only the inPoly region**
ncid = netcdf.open(geoFilePath, 'WRITE');
fvcid = netcdf.inqVarID(ncid, 'GREENFRAC');
fvc_org = netcdf.getVar(ncid, fvcid);

% Loop through grid cells and replace GVF values
for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            fvc_org(i,j,:) = GVF_grid(i,j,:); % Replace GVF values
        end
    end
end

% Write the modified GVF data back to the file
netcdf.putVar(ncid, fvcid, fvc_org);
netcdf.close(ncid);

disp('Detrended NDVI has been successfully written into the geo file')
