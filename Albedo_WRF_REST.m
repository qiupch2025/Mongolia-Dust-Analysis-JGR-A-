% Apply growth rates by land-use type to modify ALBEDO values

%% Read .mat file containing growth rate data
matFilePath = 'albedo_landUse_mean_growthRate_im.mat';
load(matFilePath, 'growth_rate_landUse'); % Dimensions: (19 land-use types, 12 months)

%% Read geo file
disp('Reading geo file')
geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/ideal_new/geo_em.d01_ideal_new.nc';
ncid = netcdf.open(geoFilePath, 'WRITE');

% Read variable ID
albedo_id = netcdf.inqVarID(ncid, 'ALBEDO12M');
albedo_org = netcdf.getVar(ncid, albedo_id);
targetLonGrid = ncread(geoFilePath, 'CLONG');
targetLatGrid = ncread(geoFilePath, 'CLAT');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

% Read land-use type
landUseType = ncread(geoFilePath, 'LU_INDEX');

% Read polygon data
s = shaperead('/home/qiupch2023/data/shp/world/world.shp');
inPoly = inpolygon(targetLonGrid, targetLatGrid, s(47).X, s(47).Y);

%% Define year range
num_years = 2023 - 2005 + 1;
months_to_modify = 1:12; % Modify all 12 months

% Compute mean growth rate for other land-use types
all_types = 1:19;
specific_types = [5, 10, 16];
other_types = setdiff(all_types, specific_types);
other_mean_growth_rate = nanmean(growth_rate_landUse(other_types, months_to_modify), 1);

growth_rate_landUse(10, 3:5)*100
growth_rate_landUse(16, 3:5)*100
growth_rate_landUse(5, 3:5)*100
other_mean_growth_rate(3:5)*100

%% Loop through grid cells and modify ALBEDO values based on land-use type
disp('Applying land-use-based growth rates to modify ALBEDO values')

for i = 1:nx-1
    for j = 1:ny-1
        if inPoly(i,j)
            lu_type = landUseType(i,j);
            for month = months_to_modify
                if ismember(lu_type, specific_types)
                    growth_rate = growth_rate_landUse(lu_type, month);
                else
                    growth_rate = other_mean_growth_rate(month);
                end
                initial_albedo = albedo_org(i,j,month);
                albedo_org(i,j,month) = initial_albedo * (1 + growth_rate * num_years);
            end
        end
    end
end

%% Write modified ALBEDO data
disp('Writing modified ALBEDO data')
netcdf.putVar(ncid, albedo_id, albedo_org);

% Close netCDF file
netcdf.close(ncid);
disp('ALBEDO growth rate update completed')
