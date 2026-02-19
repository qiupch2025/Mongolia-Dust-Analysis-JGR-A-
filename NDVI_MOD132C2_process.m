% clc;
% clear all;

% Set the folder path
ndviFolderPath = '/data/groups/lzu_public/home/qiupch2023/lustre_data/MODIS/MOD13C2/';

% Get all HDF file paths
ndviFiles = dir(fullfile(ndviFolderPath, 'MOD13C2.A*.hdf'));

% Define grid longitude and latitude
lon = linspace(-180, 179.95, 7200);
lat = linspace(90, -89.95, 3600);
[lon_highRes, lat_highRes] = meshgrid(lon, lat);

% Initialize variables
years = 2001:2023;  % Year range
n_years = length(years);
n_months = 12;  % 12 months per year
ndviData = nan(length(lat), length(lon), n_years, n_months);

% Loop through files and read data
for k = 1:length(ndviFiles)
    filename = ndviFiles(k).name;
    
    % Extract year and month
    year_str = filename(10:13);
    day_of_year = str2double(filename(14:16));
    date = datetime(str2double(year_str), 1, 1) + days(day_of_year - 1);
    year1 = year(date);
    monthIndex = month(date);  % Month index
    
    % Check whether the year is within the range
    year_idx = find(years == year1);
    if isempty(year_idx)
        continue; % Skip files outside the year range
    end
    
    % Read NDVI data
    ndviFile = fullfile(ndviFolderPath, filename);

    ndvi = hdfread(ndviFile, 'CMG 0.05 Deg Monthly NDVI'); % Modify to the actual NDVI variable name
    filename
    % Store the data into the matrix
    ndviData(:, :, year_idx, monthIndex) = double(ndvi)/10000;
    
end
% ndviData(ndviData==-0.3)=0;

[lats, lons, ~, ~] = size(ndviData);

% Create NetCDF file to save the results
outputFile = 'NDVI_month_MOD13C2_2001_2023.nc';

% Create dimensions
nccreate(outputFile, 'latitude', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'longitude', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'years', 'Dimensions', {'years', n_years}, 'Datatype', 'double');

% Create NDVI variable
nccreate(outputFile, 'ndvi_MOD13C2', 'Dimensions', {'lat', lats, 'lon', lons,'years', n_years,'months', n_months}, 'Datatype', 'double');

% Write data
ncwrite(outputFile, 'latitude', lat_highRes);
ncwrite(outputFile, 'longitude', lon_highRes);
ncwrite(outputFile, 'years', years);
ncwrite(outputFile, 'ndvi_MOD13C2', ndviData);

disp('NDVI data processing completed, and the results have been saved to the NetCDF file');
