load('NDC2k_test.mat');
normalize_data = data(:, 1: end - 1);
labels = data(:, end);
colmin = min(normalize_data);
colmax = max(normalize_data);

size(normalize_data)

for i = 1: size(normalize_data, 2)
	for j = 1: size(normalize_data, 1)
		normalize_data(j, i) = (normalize_data(j, i) - colmin(i))/(colmax(i) - colmin(i));
	end
end

% normalize_data = rescale(normalize_data, 'InputMin', colmin, 'InputMax', colmax);
data = [normalize_data labels];

save('NDC2k_test_normalized.mat', 'data');