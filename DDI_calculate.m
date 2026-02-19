% clc;
% clear all;

% Read NDVI and Albedo data
ncFile = '/home/qiupch2023/data/MODIS/data_month/ndvi_albedo_monthly_2005_2023.nc';
ndviStack = ncread(ncFile, 'ndvi');
albedoStack = ncread(ncFile, 'albedo');
lat = ncread(ncFile, 'latitude');
lon = ncread(ncFile, 'longitude');

% Read polygon data
s = shaperead('/home/qiupch2023/data/shp/world/world.shp');

% Precompute whether all points are within the polygon
inPoly = inpolygon(lon, lat, s(47).X, s(47).Y);

% Initialize NDVI and Albedo max/min matrices
ndviMaxStack = NaN(size(ndviStack, 1), size(ndviStack, 2), size(ndviStack, 3));
albedoMinStack = NaN(size(albedoStack, 1), size(albedoStack, 2), size(albedoStack, 3));

% Calculate annual NDVI maximum and Albedo minimum and apply mask
for yearIdx = 1:size(ndviStack, 3)
    % Calculate NDVI maximum
    ndviMax = max(ndviStack(:, :, yearIdx, :), [], 4, 'omitnan');
    % Calculate Albedo minimum
    albedoMin = min(albedoStack(:, :, yearIdx, :), [], 4, 'omitnan');
    
    % Apply inPoly mask: keep values inside polygon, set others to NaN
    ndviMaxStack(:, :, yearIdx) = ndviMax .* inPoly;
    albedoMinStack(:, :, yearIdx) = albedoMin .* inPoly;
end

% Calculate global min and max for NDVI and Albedo
ndviGlobalMax = max(ndviMaxStack(:), [], 'omitnan');
ndviGlobalMin = min(ndviMaxStack(:), [], 'omitnan');
albedoGlobalMax = max(albedoMinStack(:), [], 'omitnan');
albedoGlobalMin = min(albedoMinStack(:), [], 'omitnan');

fprintf('NDVI max = %.5f\n', ndviGlobalMax);
fprintf('NDVI min = %.5f\n', ndviGlobalMin);
fprintf('Albedo max = %.5f\n', albedoGlobalMax);
fprintf('Albedo min = %.5f\n', albedoGlobalMin);

% Normalize NDVI and Albedo for each year
ndviNormalizedStack = (ndviMaxStack - ndviGlobalMin) / (ndviGlobalMax - ndviGlobalMin);
albedoNormalizedStack = (albedoMinStack - albedoGlobalMin) / (albedoGlobalMax - albedoGlobalMin);

% Randomly select points within inPoly region
[rowIdx, colIdx] = find(inPoly);  % Find all points within the polygon
numPoints = 100;  % Number of randomly selected points
randomIndices = randperm(length(rowIdx), numPoints);  % Randomly select points
selectedRows = rowIdx(randomIndices);
selectedCols = colIdx(randomIndices);

% Initialize variables to store NDVI, Albedo, and coordinates for selected points
ndviSelectedPoints = [];
albedoSelectedPoints = [];
selectedLatLon = [];

for i = 1:numPoints
    row = selectedRows(i);
    col = selectedCols(i);
    
    % Save latitude and longitude of the point from 2D matrices
    selectedLatLon = [selectedLatLon; lat(row, col), lon(row, col)];
    
    % Save NDVI and Albedo values for all years at this point
    ndviSelectedPoints(:, i) = squeeze(ndviNormalizedStack(row, col, :));
    albedoSelectedPoints(:, i) = squeeze(albedoNormalizedStack(row, col, :));
end

% Output coordinates and all-year NDVI/Albedo values for selected points
fprintf('Coordinates of selected points:\n');
disp(selectedLatLon);
fprintf('NDVI values of selected points:\n');
disp(ndviSelectedPoints);
fprintf('Albedo values of selected points:\n');
disp(albedoSelectedPoints);

% Fit model using all-year data of selected points
ndviAllYears = ndviSelectedPoints(:);
albedoAllYears = albedoSelectedPoints(:);

% Linear fitting
fitType = fittype('poly1');
[fitResult, gof] = fit(ndviAllYears, albedoAllYears, fitType);

% Plot scatter and fitted line
figure;
scatter(ndviAllYears, albedoAllYears, 10, 'filled');
hold on;
xFit = linspace(min(ndviAllYears), max(ndviAllYears), 100);
yFit = fitResult(xFit);
plot(xFit, yFit, 'r-', 'LineWidth', 2);
xlabel('Normalized NDVI');
ylabel('Normalized Albedo');
title('Scatter Plot with Fit');
legend('Data Points', 'Fit Line');
grid on;

% Save figure
saveas(gcf, 'ndvi_albedo_fit.png');

% Get slope from fitting result
slope = fitResult.p1;
fprintf('Fitted slope = %.5f\n', slope);
k = -1 / slope;  % k is the negative reciprocal of the slope
fprintf('DDI coefficient = %.5f\n', k);

% Calculate DDI index for each year
ddiStack = [];

for yearIdx = 1:size(ndviNormalizedStack, 3)
    ndviNormalized = ndviNormalizedStack(:, :, yearIdx);
    albedoNormalized = albedoNormalizedStack(:, :, yearIdx);
    
    % Calculate DDI index for current year
    ddiIndex = k * ndviNormalized - albedoNormalized;
    
    % Store DDI index
    ddiStack(:, :, yearIdx) = ddiIndex;
end

% Fix latitude limits issue in georefcells
latlim = [min(lat(:)) max(lat(:))];  % Latitude range
lonlim = [min(lon(:)) max(lon(:))];  % Longitude range
R = georefcells(latlim, lonlim, size(ddiStack(:, :, 1)));

% Save yearly DDI index as TIFF files
for yearIdx = 1:size(ddiStack, 3)
    data = flipud(ddiStack(:, :, yearIdx));  % Flip vertically (latitude direction)
    tifFilename = fullfile('/home/qiupch2023/data/MODIS/DDI_end/', ['ddi_', num2str(2005 + yearIdx - 1), '.tif']);
    geotiffwrite(tifFilename, single(data), R);
    fprintf('Converted and saved file: %s\n', tifFilename);
end

% Initialize trend and significance matrices
trendMatrix = NaN(size(ddiStack, 1), size(ddiStack, 2));  % Store slopes
pValueMatrix = NaN(size(ddiStack, 1), size(ddiStack, 2));  % Store p-values

years = 1 + (1:size(ddiStack, 3)) - 1;  % Year indices

% Traverse each pixel to calculate trend and significance
for row = 1:size(ddiStack, 1)
    for col = 1:size(ddiStack, 2)
        ddiValues = squeeze(ddiStack(row, col, :));  % Extract time series
        if ~all(isnan(ddiValues))  % Skip if all values are NaN
            % Fit linear regression model
            mdl = fitlm(years', ddiValues);
            trendMatrix(row, col) = mdl.Coefficients.Estimate(2);  % Slope (2nd coefficient)
            pValueMatrix(row, col) = mdl.Coefficients.pValue(2);  % p-value (2nd coefficient)
        end
    end
end

% Save trend matrix as TIFF
trendData = flipud(trendMatrix);  % Flip vertically (latitude direction)
tifTrendFilename = fullfile('/home/qiupch2023/data/MODIS/DDI_end/', 'ddi_trend.tif');
geotiffwrite(tifTrendFilename, single(trendData), R);
fprintf('Converted and saved slope file: %s\n', tifTrendFilename);

% Save significance (p-value) matrix as TIFF
pValueData = flipud(pValueMatrix);  % Flip vertically (latitude direction)
tifPValueFilename = fullfile('/home/qiupch2023/data/MODIS/DDI_end/', 'ddi_pvalue.tif');
geotiffwrite(tifPValueFilename, single(pValueData), R);
fprintf('Converted and saved p-value file: %s\n', tifPValueFilename);

% Save all-year NDVI and Albedo values of selected points to nc file
outputNcFile = '/home/qiupch2023/data/MODIS/DDI_end/selected_points_data.nc';

% Create dimensions (number of points and number of years)
nccreate(outputNcFile, 'lat', 'Dimensions', {'points', numPoints});
nccreate(outputNcFile, 'lon', 'Dimensions', {'points', numPoints});
nccreate(outputNcFile, 'years', 'Dimensions', {'years', size(ndviSelectedPoints, 1)});
nccreate(outputNcFile, 'ndvi', 'Dimensions', {'points', numPoints, 'years', size(ndviSelectedPoints, 1)});
nccreate(outputNcFile, 'albedo', 'Dimensions', {'points', numPoints, 'years', size(ndviSelectedPoints, 1)});

% Write data
ncwrite(outputNcFile, 'lat', selectedLatLon(:, 1));
ncwrite(outputNcFile, 'lon', selectedLatLon(:, 2));
ncwrite(outputNcFile, 'years', years);
ncwrite(outputNcFile, 'ndvi', ndviSelectedPoints');
ncwrite(outputNcFile, 'albedo', albedoSelectedPoints');

fprintf('NDVI and Albedo values have been saved to file: %s\n', outputNcFile);
