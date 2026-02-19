% clc;
% clear all;

% 设置文件路径
laiFolderPath = '/data/groups/g1600002/home/qiupch2023/lustre_data/EST_2/N_L_A_trend/Inner_Mongolia/';
laiFile = fullfile(laiFolderPath, 'lai_monthly_max_2001_2023_Inner_Mongolia.nc');

% 读取维度信息
lat = ncread(laiFile, 'latitude');
lon = ncread(laiFile, 'longitude');
years = 2005:2023;
n_years = length(years);

% 读取LAI最大值数据（lat, lon, year, month），选择2005–2023年
laiData = ncread(laiFile, 'lai_max');
laiData = laiData(:, :, 5:23, :);  % 第5~23年对应2005~2023

% 读取内蒙古边界
s = shaperead('/home/qiupch2023/data/shp/Inner_Mongolia/Inner_Mongolia.shp');
inPoly = inpolygon(lon, lat, s.X, s.Y);

disp('数据读取成功');

% 春季（3–5月）平均值计算
[lats, lons, ~, ~] = size(laiData);
lai_spring = nan(lats, lons, n_years);

for y = 1:n_years
    lai_spring(:, :, y) = mean(laiData(:, :, y, 3:5), 4, 'omitnan');
end

disp('春季平均值计算完成');

% 初始化趋势图和p值图
lai_trend_map = nan(lats, lons);
lai_pvalue_map = nan(lats, lons);

for r = 1:lats
    for c = 1:lons
        if inPoly(r, c)
            series = squeeze(lai_spring(r, c, :));
            if all(~isnan(series))
                X = [ones(n_years, 1), years'];
                [b, ~, ~, ~, stats] = regress(series, X);
                lai_trend_map(r, c) = b(2);     % 斜率
                lai_pvalue_map(r, c) = stats(3); % p值
            end
        end
    end
end

disp('春季像素趋势计算完成');

% 区域平均趋势
spring_mean_series = nan(n_years, 1);
for y = 1:n_years
    temp = lai_spring(:, :, y);
    spring_mean_series(y) = mean(temp(inPoly), 'omitnan');
end
[b_mean, ~, ~, ~, stats_mean] = regress(spring_mean_series, [ones(n_years, 1), years']);
lai_mean_trend = b_mean(2);
lai_mean_pvalue = stats_mean(3);

disp('整体平均趋势计算完成');

% 创建并写入 NetCDF 文件
outputFile = 'LAI_spring_trend_Inner_Mongolia.nc';

nccreate(outputFile, 'latitude', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'longitude', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'years', 'Dimensions', {'years', n_years}, 'Datatype', 'double');

nccreate(outputFile, 'lai_trend', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'lai_pvalue', 'Dimensions', {'lat', lats, 'lon', lons}, 'Datatype', 'double');
nccreate(outputFile, 'lai_mean_trend', 'Dimensions', {'scalar', 1}, 'Datatype', 'double');
nccreate(outputFile, 'lai_mean_pvalue', 'Dimensions', {'scalar', 1}, 'Datatype', 'double');
nccreate(outputFile, 'lai_spring_mean_series', 'Dimensions', {'years', n_years}, 'Datatype', 'double');
ncwrite(outputFile, 'lai_spring_mean_series', spring_mean_series);
% 写入数据
ncwrite(outputFile, 'latitude', lat);
ncwrite(outputFile, 'longitude', lon);
ncwrite(outputFile, 'years', years);
ncwrite(outputFile, 'lai_trend', lai_trend_map);
ncwrite(outputFile, 'lai_pvalue', lai_pvalue_map);
ncwrite(outputFile, 'lai_mean_trend', lai_mean_trend);
ncwrite(outputFile, 'lai_mean_pvalue', lai_mean_pvalue);

disp('NetCDF 写入完成');
