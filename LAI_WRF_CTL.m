% clc;
% clear;

% Read LAI data for 2005–2023
disp('Reading LAI data...')
laiData = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/lai_monthly_2005_2023.nc','lai');
lon_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/lai_monthly_2005_2023.nc', 'longitude');
lat_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/lai_monthly_2005_2023.nc', 'latitude');

% Time dimension
years = 2005:2023;
num_years = length(years);
[lats, lons, ~, num_months] = size(laiData); % Correct dimension order

% Initialize storage for detrended 2023 data
lai_detrended_2023 = nan(lats, lons, num_months);

disp('Removing long-term LAI trends for 2005–2023 while preserving seasonal variability')

% Loop over each grid cell
for r = 1:lats
    disp(['Processing row: ', num2str(r)]) % Display progress
    for c = 1:lons
        for month = 1:num_months
            % Extract the LAI series (2005–2023) for the current grid cell and month
            laiSeries = squeeze(laiData(r, c, :, month)); % Correct dimension order
            
            % Ensure data are valid
            if all(~isnan(laiSeries))
                % Least-squares linear trend fit: y = a*t + b
                p_lai = polyfit(years, laiSeries, 1);
                
                % Compute the trend component at 2023
                trend_lai_at_2023 = p_lai(1) * (2023 - 2005);
                
                % Detrend
                lai_detrended_2023(r, c, month) = laiData(r, c, end, month) - trend_lai_at_2023;
            end
        end
    end
end

disp('Detrended LAI computation completed')

%% Read target grid
geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2005/geo_em.d01_2005.nc';
targetLonGrid = ncread(geoFilePath,'CLONG');
targetLatGrid = ncread(geoFilePath,'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

% Initialize LAI for the 119x99 grid
lai_mean = zeros(nx-1, ny-1, num_months);

% Read polygon data
s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

% Determine which points are inside the polygon
inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);

disp('Computing mean detrended LAI within each 119x99 grid cell, computed only inside inPoly')

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
                sub_data = lai_detrended_2023(:,:,month);
                sub_data = sub_data(mask);
                
                % Compute mean LAI
                lai_mean(i,j,month) = nanmean(sub_data, 'all');
            end
        end
    end
end

lai_mean(lai_mean<0) = 0; % Ensure LAI is non-negative

disp('Mean detrended LAI computed; writing to geo file')

%% Replace LAI data in the geo file
ncid = netcdf.open(geoFilePath, 'WRITE');
laiid = netcdf.inqVarID(ncid, 'LAI12M');
lai_org = netcdf.getVar(ncid, laiid);

% Loop through grid cells and replace LAI values
for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            lai_org(i,j,:) = lai_mean(i,j,:); % Replace LAI values
        end
    end
end

% Write the modified LAI data back to the file
netcdf.putVar(ncid, laiid, lai_org);
netcdf.close(ncid);

disp('Detrended LAI has been successfully written into the geo file')
