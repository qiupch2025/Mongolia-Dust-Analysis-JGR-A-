% clc;
% clear all;

% 设置文件夹路径
ndviFolderPath = '/data/groups/lzu_public/home/qiupch2023/lustre_data/MODIS/MOD13C2/';
ndviFiles = dir(fullfile(ndviFolderPath, 'MOD13C2.A*.hdf'));

% 定义网格经纬度
lon = linspace(-180, 179.95, 7200);
lat = linspace(90, -89.95, 3600);
[lon_highRes, lat_highRes] = meshgrid(lon, lat);

% 初始化变量
years = 2005:2023;
n_years = length(years);
ndviData = nan(length(lat), length(lon), n_years, 12);

% 遍历文件并读取数据
for k = 1:length(ndviFiles)
    filename = ndviFiles(k).name;
    
    % 提取时间信息
    year_str = filename(10:13);
    day_of_year = str2double(filename(14:16));
    date = datetime(str2double(year_str), 1, 1) + days(day_of_year - 1);
    year1 = year(date);
    monthIndex = month(date);
    
    % 仅保留目标年份
    year_idx = find(years == year1);
    if isempty(year_idx)
        continue;
    end

    % 读取 NDVI 数据
    ndviFile = fullfile(ndviFolderPath, filename);
    ndvi = hdfread(ndviFile, 'CMG 0.05 Deg Monthly NDVI');
    ndvi = double(ndvi) / 10000;
    ndvi(ndvi == -0.3) = 0;
    
    ndviData(:, :, year_idx, monthIndex) = ndvi;
end

% 读取蒙古边界
s = shaperead('/home/qiupch2023/data/shp/Mongolia/Mongolia.shp');
inPoly = inpolygon(lon_highRes, lat_highRes, s.X, s.Y);

disp('数据读取成功');

% 计算春季（3–5月）NDVI平均
[lats, lons, ~, ~] = size(ndviData);
ndvi_spring = nan(lats, lons, n_years);
for y = 1:n_years
    ndvi_spring(:, :, y) = mean(ndviData(:, :, y, 3:5), 4, 'omitnan');
end

disp('春季 NDVI 平均计算完成');

% 计算趋势和 p 值
ndvi_trend_map = nan(lats, lons);
ndvi_pvalue_map = nan(lats, lons);

for r = 1:lats
    for c = 1:lons
        if inPoly(r, c)
            series = squeeze(ndvi_spring(r, c, :));
            if all(~isnan(series))
                X = [ones(n_years, 1), years'];
                [b, ~, ~, ~, stats] = regress(series, X);
                ndvi_trend_map(r, c) = b(2);
                ndvi_pvalue_map(r, c) = stats(3);
            end
        end
    end
end

disp('春季趋势计算完成');

% 区域平均趋势
spring_mean_series = nan(n_years, 1);
for y = 1:n_years
    tmp = ndvi_spring(:, :, y);
    spring_mean_series(y) = mean(tmp(inPoly), 'omitnan');
end
[b_mean, ~, ~, ~, stats_mean] = regress(spring_mean_series, [ones(n_years, 1), years']);
ndvi_mean_trend = b_mean(2);
ndvi_mean_pvalue = stats_mean(3);

disp('区域平均趋势计算完成');

% 创建 NetCDF 文件
outputFile = 'NDVI_spring_trend_Mongolia.nc';

nccreate(outputFile, 'latitude', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'longitude', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'years', 'Dimensions', {'years', n_years}, 'Datatype', 'double');

nccreate(outputFile, 'ndvi_trend', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'ndvi_pvalue', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'ndvi_mean_trend', 'Dimensions', {'scalar', 1}, 'Datatype', 'double');
nccreate(outputFile, 'ndvi_mean_pvalue', 'Dimensions', {'scalar', 1}, 'Datatype', 'double');
nccreate(outputFile, 'ndvi_spring_mean_series', 'Dimensions', {'years', n_years}, 'Datatype', 'double');
ncwrite(outputFile, 'ndvi_spring_mean_series', spring_mean_series);
% 写入数据
ncwrite(outputFile, 'latitude', lat_highRes);
ncwrite(outputFile, 'longitude', lon_highRes);
ncwrite(outputFile, 'years', years);
ncwrite(outputFile, 'ndvi_trend', ndvi_trend_map);
ncwrite(outputFile, 'ndvi_pvalue', ndvi_pvalue_map);
ncwrite(outputFile, 'ndvi_mean_trend', ndvi_mean_trend);
ncwrite(outputFile, 'ndvi_mean_pvalue', ndvi_mean_pvalue);

disp('NDVI春季趋势数据处理完成，结果已保存至NetCDF文件');
