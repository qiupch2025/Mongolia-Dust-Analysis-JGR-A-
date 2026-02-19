clc
clear

disp('Start reading geo file and shapefile')

% Path to geo file
geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/Mongolia_source/geo/geo_em.d01_2023_erod0.nc';

% Read latitude and longitude information
lat_grid = ncread(geoFilePath, 'CLAT');
lon_grid = ncread(geoFilePath, 'CLONG');
size(lat_grid)

% Read shapefile
s = shaperead('/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/shp/mongolia/menggu.shp');

% Extract Mongolia polygon (assuming the 47th is Mongolia)
mongolia = s;

% Determine whether grid points are within Mongolia
disp('Determining Mongolia region...')
inPoly = inpolygon(lon_grid, lat_grid, mongolia.X, mongolia.Y);
size(inPoly)
sum(inPoly,'all')

% Read EROD variable
disp('Reading EROD variable...')
erod_data = ncread(geoFilePath, 'EROD');
size(erod_data)

% Set EROD within Mongolia to 0
for k = 1:size(erod_data,3)
    tmp = erod_data(:,:,k);  % Extract the k-th layer (2D)
    tmp(inPoly) = 0;         % Assign 0 using 2D logical indexing
    erod_data(:,:,k) = tmp;  % Write back
end

size(erod_data)

% Write modified EROD variable back to the geo file
disp('Writing modified EROD variable...')
ncid = netcdf.open(geoFilePath, 'WRITE');
erod_id = netcdf.inqVarID(ncid, 'EROD');
erod_org = netcdf.getVar(ncid, erod_id);
size(erod_org)

% Traverse grid points and replace EROD values
for i = 1:size(erod_org,1)
    for j = 1:size(erod_org,2)
        if inPoly(i, j)
            erod_org(i,j,:) = erod_data(i,j,:); % Replace EROD values
        end
    end
end
netcdf.putVar(ncid, erod_id, erod_org);
netcdf.close(ncid);

disp('✅ EROD values within Mongolia have been set to 0')
