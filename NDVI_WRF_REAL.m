% clc
% clear

% lon=linspace(-180,179.95,7200);
% lat=linspace(90,-89.95,3600);
% [lon_highRes, lat_highRes] = meshgrid(lon, lat);

NDVIS = 0.05;
NDVIV = 0.90;
disp('Starting to read data')
ndviData = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'ndvi_MOD13C2');
ndvi_data = squeeze(ndviData(:, :, end, :)); % Replace with your NDVI data file
lon_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'longitude');
lat_highRes = ncread('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_DATA/NDVI_ALBEDO_LAI/NDVI_month_MOD13C2.nc', 'latitude');

% Remove outliers and NDVI values over water
ndvi_data(ndvi_data == -0.3) = 0; % Set NDVI == -0.3 as water and replace with 0

% Compute Green Vegetation Fraction (GVF)
GVF = (ndvi_data - NDVIS) ./ (NDVIV - NDVIS);

% Constrain GVF to the range [0, 1]
GVF(GVF < 0) = 0;
GVF(GVF > 1) = 1;

% Display results
% figure;
% imagesc(GVF);
% colorbar;
% title('2023 Green Vegetation Fraction (GVF)');

% geo file path
geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2023/geo_em.d01_2023.nc';
% Read target grid definition and lat/lon information
targetLonGrid = ncread(geoFilePath,'CLONG');
targetLatGrid = ncread(geoFilePath,'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

% Initialize vegetation cover area ratio matrix (119x99 grid)
fvc_mean = zeros(nx-1, ny-1, size(GVF,3));

% Read polygon data
s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

% Precompute whether all points are inside the polygon
inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);
disp('Data read successfully, start computing the average vegetation cover for each grid in the Mongolia region')
% Compute the average vegetation cover within each 119x99 grid cell
for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            % Get the boundaries of the current 119x99 grid cell
            lat_min = min([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lat_max = max([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lon_min = min([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);
            lon_max = max([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);
    
            % Extract high-resolution data within the current grid cell
            mask = (lat_highRes >= lat_min) & (lat_highRes <= lat_max) & (lon_highRes >= lon_min)...
                & (lon_highRes <= lon_max);
            for month = 1:size(GVF,3)
                GVF_month = GVF(:,:,month);
                sub_data = GVF_month(mask);
                fvc_mean(i,j,month) = nanmean(sub_data,'all');
            end
        end
    end
end

disp('Computation completed, starting to replace geo')
% Replace vegetation cover in the geo file
ncid = netcdf.open(geoFilePath, 'WRITE');
% Read variable ID
fvcid = netcdf.inqVarID(ncid, 'GREENFRAC');
fvc_org = netcdf.getVar(ncid, fvcid);

% Iterate over each grid cell and modify vegetation cover
for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i, j)
            fvc_org(i,j,:) = fvc_mean(i,j,:);
        end
    end
end
% Write the replaced data back to the file
netcdf.putVar(ncid, fvcid, fvc_org);
% Close the netCDF file
netcdf.close(ncid);
disp('NDVI replacement completed')
