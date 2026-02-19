% clc;
% clear;

% Read Albedo data from 2005–2023
disp('Reading data...')
albedoData = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'albedo');
lon_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'longitude');
lat_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'latitude');

% Time dimension
years = 2005:2023;
num_years = length(years);
[lats, lons, ~, num_months] = size(albedoData); % Correct dimension order

% Initialize array for detrended 2023 Albedo
albedo_detrended_2023 = nan(lats, lons, num_months);

disp('Removing long-term Albedo trends (2005–2023) while preserving seasonal variability')

% Loop through each grid cell
for r = 1:lats
    disp(['Processing row: ', num2str(r)]) % Display progress
    for c = 1:lons
        for month = 1:num_months
            % Extract Albedo series for this grid and month (2005–2023)
            albedoSeries = squeeze(albedoData(r, c, :, month)); % Correct dimension order
            
            % Find non-NaN indices
            valid_idx = ~isnan(albedoSeries);

            % Require at least two valid points to fit trend
            if sum(valid_idx) >= 2
                % Fit linear trend using non-NaN data only
                p_albedo = polyfit(years(valid_idx), albedoSeries(valid_idx), 1);
                
                % Calculate trend component for 2023
                trend_albedo_at_2023 = p_albedo(1) * (2023 - 2005);
                
                % Detrend
                albedo_detrended_2023(r, c, month) = albedoData(r, c, end, month) - trend_albedo_at_2023;
            else
                % Not enough valid data points → assign NaN
                albedo_detrended_2023(r, c, month) = NaN;
            end
        end
    end
end

disp('Detrended Albedo computation completed')

%% Read target grid
geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2005/geo_em.d01_2005.nc';
targetLonGrid = ncread(geoFilePath,'CLONG');
targetLatGrid = ncread(geoFilePath,'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

% Initialize Albedo array for 119x99 grid
albedo_mean = zeros(nx-1, ny-1, num_months);

% Read polygon data
s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

% Determine which grid cells are inside the polygon
inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);

disp('Calculating mean detrended Albedo within each 119x99 grid cell, restricted to inPoly region')

% Loop through target grid
for i = 1:nx-1
    i
    for j = 1:ny-1
        if inPoly(i, j)
            % Get boundaries of current 119x99 grid cell
            lat_min = min([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lat_max = max([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lon_min = min([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);
            lon_max = max([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);

            % Extract high-resolution data within the current cell
            mask = (lat_highRes >= lat_min) & (lat_highRes <= lat_max) & ...
                   (lon_highRes >= lon_min) & (lon_highRes <= lon_max);

            for month = 1:num_months
                sub_data = albedo_detrended_2023(:,:,month);
                sub_data = sub_data(mask);
                
                % Compute mean Albedo
                albedo_mean(i,j,month) = nanmean(sub_data, 'all');
            end
        end
    end
end

disp('Mean detrended Albedo computed; writing to geo file')

%% Replace Albedo data in the geo file
ncid = netcdf.open(geoFilePath, 'WRITE');
albedo_id = netcdf.inqVarID(ncid, 'ALBEDO12M');
albedo_org = netcdf.getVar(ncid, albedo_id);

% Loop through grid cells and replace Albedo values
for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            albedo_org(i,j,:) = albedo_mean(i,j,:)*100; % Replace Albedo values
        end
    end
end

% Write modified Albedo data back to file
netcdf.putVar(ncid, albedo_id, albedo_org);
netcdf.close(ncid);

disp('Detrended Albedo successfully replaced in geo file')
