% Load the MNIST data 
[XTrain, YTrain] = digitTrain4DArrayData; % Training data
[XTest, YTest] = digitTest4DArrayData;   % Test data

% Function to create a convolution block
function layers = convBlock(filterSize, numFilters)
    layers = [
        convolution2dLayer(filterSize, numFilters, 'Padding', 'same', 'Stride', 1)
        batchNormalizationLayer
        reluLayer
    ];
end

% Define CNN Architecture
layers = [
    imageInputLayer([28 28 1], 'Name', 'Input')
    
    % Add convolution blocks
    convBlock(3, 8)
    maxPooling2dLayer(2, 'Stride', 2)
  
    convBlock(3, 16)
    maxPooling2dLayer(2, 'Stride', 2)

    convBlock(3, 32)
    maxPooling2dLayer(2, 'Stride', 2)

    convBlock(3, 64)
    maxPooling2dLayer(2, 'Stride', 2)

    convBlock(3, 128)
    %dropoutLayer(0.4)
    
    % Fully connected and output layers
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% Visualize the network
%lgraph = layerGraph(layers);
%plot(lgraph);

% Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest, YTest}, ...
    'ValidationFrequency', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress'); 

% Train the network
net = trainNetwork(XTrain, YTrain, layers, options);

% Predict labels for test data
YPred = classify(net, XTest);

% Calculate accuracy
accuracy = mean(YPred == YTest);
disp(['Test Accuracy: ', num2str(accuracy)]);

% Visualize predictions (modularized)
function visualizePredictions(XTest, YPred, numImages)
    idx = randperm(size(XTest, 4), numImages); % Randomly select test images
    figure;
    for i = 1:numImages
        subplot(sqrt(numImages), sqrt(numImages), i);
        imshow(XTest(:, :, :, idx(i)));
        title(string(YPred(idx(i))));
    end
end

% Call visualization for 16 test images
visualizePredictions(XTest, YPred, 16);

% Save the trained network
save('CNN_new.mat', 'net');
