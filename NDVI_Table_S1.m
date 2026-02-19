% Full code: from reading to computing trend growth rate (NDVI)

%% Path definitions (replace with your actual paths)
ndviFolderPath = '/data/groups/lzu_public/home/qiupch2023/lustre_data/MODIS/MOD13C2/';
geoFilePath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/geo/shao04/2023/geo_em.d01_2023.nc';
shpPath = '/home/qiupch2023/data/shp/Mongolia/Mongolia.shp';
saveFileName = 'ndvi_landUse_mean_growthRate_m.mat';

%% Define high-resolution latitude/longitude
lon = linspace(-180, 179.95, 7200);
lat = linspace(90, -89.95, 3600);
[lon_highRes, lat_highRes] = meshgrid(lon, lat);

%% Read NDVI data
ndviFiles = dir(fullfile(ndviFolderPath, 'MOD13C2.A*.hdf'));
years = 2005:2023;
n_years = length(years);
months = 12;
ndviData = nan(3600, 7200, n_years, months);

for k = 1:length(ndviFiles)
    filename = ndviFiles(k).name;
    year_str = filename(10:13);
    day_of_year = str2double(filename(14:16));
    date = datetime(str2double(year_str), 1, 1) + days(day_of_year - 1);
    year_idx = find(years == year(date));
    month_idx = month(date);
    if isempty(year_idx)
        continue;
    end

    ndviFile = fullfile(ndviFolderPath, filename);
    ndvi = hdfread(ndviFile, 'CMG 0.05 Deg Monthly NDVI');
    filename
    ndviData(:,:,year_idx,month_idx) = double(ndvi)/10000;
end
ndviData(ndviData==-0.3)=0;

%% Read target grid information
targetLatGrid = ncread(geoFilePath, 'CLAT');
targetLonGrid = ncread(geoFilePath, 'CLONG');
lat_grid = ncread(geoFilePath, 'XLAT_C');
lon_grid = ncread(geoFilePath, 'XLONG_C');
[nx, ny] = size(lat_grid);

%% Read polygon and determine in-range grids
s = shaperead(shpPath);
inPoly = inpolygon(targetLonGrid, targetLatGrid, s.X, s.Y);

%% Interpolate original data to target grid
ndvi_target = nan(nx-1, ny-1, n_years, months);

for i = 1:nx-1
    disp(['Interpolation completed ', num2str(i), '/', num2str(nx-1)]);
    for j = 1:ny-1
        if inPoly(i,j)
            lat_min = min([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lat_max = max([lat_grid(i,j), lat_grid(i+1,j), lat_grid(i,j+1), lat_grid(i+1,j+1)]);
            lon_min = min([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);
            lon_max = max([lon_grid(i,j), lon_grid(i+1,j), lon_grid(i,j+1), lon_grid(i+1,j+1)]);

            mask_highRes = (lat_highRes >= lat_min & lat_highRes <= lat_max) & ...
                           (lon_highRes >= lon_min & lon_highRes <= lon_max);

            if nnz(mask_highRes) > 0
                for y = 1:n_years
                    for m = 1:months
                        data_sub = ndviData(:,:,y,m);
                        ndvi_target(i,j,y,m) = nanmean(data_sub(mask_highRes));
                    end
                end
            end
        end
    end
end

disp('Data interpolation to target grid completed');

%% Compute annual means by land-use type
landUseType = ncread(geoFilePath, 'LU_INDEX');
mean_ndvi_byLandUse = nan(19, n_years, months);

for lu = 1:19
    lu_mask = (landUseType == lu) & inPoly;
    for y = 1:n_years
        for m = 1:months
            current_data = ndvi_target(:,:,y,m);
            mean_ndvi_byLandUse(lu,y,m) = nanmean(current_data(lu_mask));
        end
    end
end

disp('Annual means by land-use type computed');

%% Compute trends and growth rates
growth_rate_landUse = nan(19, months);

for lu = 1:19
    for m = 1:months
        ndviSeries = squeeze(mean_ndvi_byLandUse(lu,:,m))';
        if all(~isnan(ndviSeries))
            X = [ones(n_years,1), years'];
            b = regress(ndviSeries, X);
            trend = b(2);
            fitted_2005 = b(1) + b(2)*2005;
            if abs(fitted_2005) > eps
                growth_rate_landUse(lu,m) = trend / fitted_2005;
            else
                growth_rate_landUse(lu,m) = NaN;
            end
        end
    end
end

disp('Trends and growth rates by land-use type computed');

%% Save results
save(saveFileName, 'mean_ndvi_byLandUse', 'growth_rate_landUse');
disp('All results have been saved');
