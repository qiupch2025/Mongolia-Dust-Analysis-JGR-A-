% clc;
% clear;

% % Define high-resolution lat/lon grid
% lon=linspace(-180,179.95,7200);
% lat=linspace(90,-89.95,3600);
% [lon_highRes, lat_highRes] = meshgrid(lon, lat);

% Read 2023 Albedo data
disp('Reading data')
albedoData = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'albedo');
albedo_data = squeeze(albedoData(:, :, end, :)); % Replace with your Albedo data file
lon_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'longitude');
lat_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/albedo_monthly_2005_2023.nc', 'latitude');

% geo file path
geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2023/geo_em.d01_2023.nc';
% Read target grid definition and lat/lon info
targetLonGrid = ncread(geoFilePath,'CLONG');
targetLatGrid = ncread(geoFilePath,'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

% Initialize Albedo matrix (119x99 grid)
albedo_mean = zeros(nx-1, ny-1, size(albedo_data,3));

% Read polygon data
s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

% Precompute whether all points are inside the polygon
inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);

% Compute mean Albedo within each 119x99 grid cell (only within inPoly)
disp('Computing mean Albedo within each 119x99 grid cell, restricted to inPoly')
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
            for month = 1:size(albedo_data,3)
                albedo_data_month = albedo_data(:,:,month);
                sub_data = albedo_data_month(mask);
                % Compute mean Albedo
                albedo_mean(i,j,month) = nanmean(sub_data, 'all');
            end
        end
    end
end
% albedo_mean(albedo_mean>0.5)=albedo_mean(albedo_mean>0.5)/2;

% Replace Albedo data in the geo file
disp('Replacing Albedo in geo file')
ncid = netcdf.open(geoFilePath, 'WRITE');
% Read variable ID; assume the Albedo variable is named 'ALBEDO12M'
albedo_id = netcdf.inqVarID(ncid, 'ALBEDO12M');
albedo_org = netcdf.getVar(ncid, albedo_id);

% Iterate through each grid cell and replace Albedo values inside inPoly
for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            albedo_org(i,j,:) = albedo_mean(i,j,:)*100; % Replace Albedo values
        end
    end
end

% Write the modified Albedo data back to the file
netcdf.putVar(ncid, albedo_id, albedo_org);
% Close the netCDF file
netcdf.close(ncid);
disp('Albedo replacement completed')
