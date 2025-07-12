function [data, mu, sigma] = standardizeByCol(data)
    [n, ~] = size(data);
    mu = mean(data, 1);
    sigma = std(data, 1);
    data = (data - repmat(mu, n, 1)) ./ repmat(sigma, n, 1);
end