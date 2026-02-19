clc;
clear all

% Read polygon data
s = shaperead('/home/qiupch2023/data/MODIS/world_shp/world.shp');

% Define grid longitude and latitude
lon = linspace(-180, 179.95, 7200);
lat = linspace(90, -89.95, 3600);
[lon_highRes, lat_highRes] = meshgrid(lon, lat);

% Precompute whether all points are within the polygon
inPoly = inpolygon(lon_highRes, lat_highRes, s(47).X, s(47).Y);

% Get all yearly folders
albedoFolders = dir('/home/qiupch2023/data/MODIS/MCD43C3/MCD43C3_*');

% Initialize a 4D matrix for Albedo
albedoMinStack = NaN(3600, 7200, length(albedoFolders), 12);  % Store monthly minimum Albedo

% Loop through all year folders
for yearIdx = 1:length(albedoFolders)
    % Process Albedo data for the current year
    albedoFiles = dir(fullfile(albedoFolders(yearIdx).folder, albedoFolders(yearIdx).name, '*.hdf'));
    
    % Extract day of year from filename and calculate month
    for k = 1:length(albedoFiles)
        filename = albedoFiles(k).name
        year_str = filename(10:13);
        day_of_year = str2double(filename(14:16));
        date = datetime(str2double(year_str), 1, 1) + days(day_of_year - 1);
        monthIndex = month(date);  % Month index

        % Read Albedo data and convert to floating-point
        albedoData_wsa = double(hdfread(fullfile(albedoFiles(k).folder, albedoFiles(k).name), 'Albedo_WSA_shortwave')) / 1000;
        albedoData_bsa = double(hdfread(fullfile(albedoFiles(k).folder, albedoFiles(k).name), 'Albedo_BSA_shortwave')) / 1000;
        u = double(hdfread(fullfile(albedoFiles(k).folder, albedoFiles(k).name), 'Local_Solar_Noon'));
        u(u == 255) = nan;  % Convert invalid values to NaN
        s = 0.122 + 0.85 * exp(-4.8 * cos(deg2rad(u)));
        albedoData = (1 - s) .* albedoData_bsa + s .* albedoData_wsa;

        % Use nanmin to update the monthly minimum Albedo
        albedoMinStack(:, :, yearIdx, monthIndex) = nanmin(albedoMinStack(:, :, yearIdx, monthIndex), albedoData);  % Update minimum value
    end
end

% Replace invalid values in Albedo with NaN
albedoMinStack(albedoMinStack == 32.7670) = NaN;  % Set invalid values to NaN if needed

% Create NetCDF file (only Albedo)
nccreate('albedo_monthly_2005_2023.nc', 'albedo', ...
    'Dimensions', {'lat', 3600, 'lon', 7200, 'year', length(albedoFolders), 'month', 12});
nccreate('albedo_monthly_2005_2023.nc', 'longitude', 'Dimensions', {'lat', 3600, 'lon', 7200});
nccreate('albedo_monthly_2005_2023.nc', 'latitude', 'Dimensions', {'lat', 3600, 'lon', 7200});

% Write data to NetCDF file
ncwrite('albedo_monthly_2005_2023.nc', 'albedo', albedoMinStack);
ncwrite('albedo_monthly_2005_2023.nc', 'longitude', lon_highRes);
ncwrite('albedo_monthly_2005_2023.nc', 'latitude', lat_highRes);

disp('Albedo data processing completed, and results have been saved to NetCDF file.');
