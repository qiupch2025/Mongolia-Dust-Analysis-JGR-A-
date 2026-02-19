% clc;
% clear;

% Define high-resolution lat/lon grid

% lon_highRes = ncread('LAI_mean_trend_with_pvalues.nc','longitude');
% lat_highRes = ncread('LAI_mean_trend_with_pvalues.nc','latitude');

% Read LAI data for 2023
disp('Reading LAI data for 2023')
laiData = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/lai_monthly_2005_2023.nc','lai');
lai_data = squeeze(laiData(:, :, end, :)); % Replace with your LAI data file
lon_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/lai_monthly_2005_2023.nc', 'longitude');
lat_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/lai_monthly_2005_2023.nc', 'latitude');

% geo file path
geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2023/geo_em.d01_2023.nc';
% Read target grid definition and lat/lon info
targetLonGrid = ncread(geoFilePath,'CLONG');
targetLatGrid = ncread(geoFilePath,'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

% Initialize LAI matrix (119x99 grid)
lai_mean = zeros(nx-1, ny-1, size(lai_data,3));

% Read polygon data
s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

% Precompute whether all points are inside the polygon
inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);

% Compute the mean LAI within each 119x99 grid cell, only inside inPoly
disp('Computing mean LAI within each 119x99 grid cell, only inside inPoly')
for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            % Get boundaries of the current 119x99 grid cell
            lat_min = min([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lat_max = max([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lon_min = min([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);
            lon_max = max([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);

            % Extract high-resolution data within the current grid cell
            mask = (lat_highRes >= lat_min) & (lat_highRes <= lat_max) & (lon_highRes >= lon_min) & (lon_highRes <= lon_max);
            for month = 1:size(lai_data,3)
                lai_data_month = lai_data(:,:,month);
                sub_data = lai_data_month(mask);
                % Compute mean LAI
                lai_mean(i,j,month) = nanmean(sub_data, 'all');
            end
        end
    end
end

lai_mean(lai_mean<0)=0;

% Replace LAI data in the geo file
ncid = netcdf.open(geoFilePath, 'WRITE');
% Read variable ID, assuming the LAI variable is 'LAI'
laiid = netcdf.inqVarID(ncid, 'LAI12M');
lai_org = netcdf.getVar(ncid, laiid);

% Iterate over each grid cell and replace LAI values inside inPoly
for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            lai_org(i,j,:) = lai_mean(i,j,:); % Replace LAI values
        end
    end
end

% Write the modified LAI data back to the file
netcdf.putVar(ncid, laiid, lai_org);
% Close the netCDF file
netcdf.close(ncid);
disp('LAI replacement completed')
