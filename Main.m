clear
clc
warning off

% 读取数据
data = readtable('汇整数据.xlsx');
features = table2array(data(:, [2, 3, 4, 5]));
target = table2array(data(:, 6));

% 数据标准化
[featuresNorm, muF, sigmaF] = standardizeByCol(features);
[targetNorm, muT, sigmaT] = standardizeByCol(target);

windowSize = 30;
nrows = size(featuresNorm, 1) - windowSize;

featuresAll = cell(nrows, 1);
for row = 1:nrows
    featuresAll{row} = featuresNorm(row:windowSize+row - 1, :).';
end
responseAll = targetNorm(windowSize+1:end);

numChannels = size(features, 2);
maxPosition = 256;
numHeads = 4;
numKeyChannels = numHeads * 32;

layers = [
    sequenceInputLayer(numChannels, Name='input')
    positionEmbeddingLayer(numChannels, maxPosition, Name='pos-emb')
    additionLayer(2, Name='add')
    selfAttentionLayer(numHeads, numKeyChannels, 'AttentionMask', 'causal')
    selfAttentionLayer(numHeads, numKeyChannels)
    indexing1dLayer('last')
    fullyConnectedLayer(1)
    regressionLayer
];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'input', 'add/in2');

maxEpochs = 300;
miniBatchSize = 64;
learningRate = 0.0095;
solver = 'adam';
shuffle = 'every-epoch';
gradientThreshold = 10;
executionEnvironment = 'auto';

options = trainingOptions(solver, ...
    'Plots', 'training-progress', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', shuffle, ...
    'InitialLearnRate', learningRate, ...
    'GradientThreshold', gradientThreshold, ...
    'ExecutionEnvironment', executionEnvironment);

train = true; 
if train
    net = trainNetwork(featuresAll, responseAll, lgraph, options);
else
    load transformer_flood_model.mat net
end

yPredAll = predict(net, featuresAll);

% 反标准化
predictAll = yPredAll * sigmaT + muT;
actualAll = responseAll * sigmaT + muT;

% ---- 绘制预测结果与真实值对比图 ----
figure;
plot(1:length(predictAll), predictAll, 'r--', 'DisplayName', 'Predicted value', 'LineWidth', 1.5);
xlabel('Day', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Flood Severity (FRF)', 'FontSize', 14, 'FontName', 'Times New Roman');
title('Flood Severity Prediction', 'FontSize', 16, 'FontWeight', 'bold');
legend('show');
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'FontName', 'Times New Roman');

% ---- 绘制预测结果与真实值的散点图 ----
figure;
scatter(actualAll, predictAll, 50, 'filled', 'MarkerFaceColor', [0.2 0.6 0.8], 'MarkerEdgeColor', [0.1 0.4 0.6]);
hold on;
xlim = [min([actualAll; predictAll]) max([actualAll; predictAll])];
ylim = xlim;
plot(xlim, ylim, 'k--', 'LineWidth', 1.5);
xlabel('True Level (FRF)', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Predicted Level (FRF)', 'FontSize', 14, 'FontName', 'Times New Roman');
title('Scatter Plot of Predicted vs True Values', 'FontSize', 16, 'FontWeight', 'bold');
legend('Predicted vs True', 'y = x', 'Location', 'Best');
grid on;
box on;
set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'FontName', 'Times New Roman');

% ---- 绘制残差图 ----
residuals = actualAll - predictAll;

% 绘制残差图：预测值 vs 残差
figure;
subplot(1, 2, 1);
scatter(predictAll, residuals, 50, 'filled', 'MarkerFaceColor', [0.2 0.6 0.8], 'MarkerEdgeColor', [0.1 0.4 0.6]);
xlabel('Predicted Value (FRF)', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Residuals', 'FontSize', 14, 'FontName', 'Times New Roman');
title('Residuals vs Predicted Values', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'FontName', 'Times New Roman');

% 绘制残差分布：直方图
subplot(1, 2, 2);
histogram(residuals, 30, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', [0.1 0.4 0.6], 'LineWidth', 1.5);
xlabel('Residuals', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Frequency', 'FontSize', 14, 'FontName', 'Times New Roman');
title('Residual Distribution', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'FontName', 'Times New Roman');
