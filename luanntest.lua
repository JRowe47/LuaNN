-- Assuming the neural network library is already included with the name 'luann'
local luann = require 'luann'

-- Configuration for the neural network
local config = {
    learningRate = 0.01,
    weightInitMethod = 'xavier',
    layers = {
        3,  -- 3 neurons in the input layer
        {numCells = 5, weightInitMethod = 'xavier'},  -- 5 neurons in the hidden layer
        {numCells = 2, weightInitMethod = 'xavier'}   -- 2 neurons in the output layer (for softmax)
    },
    dropoutRates = {0, 0.2}  -- Using dropout in the hidden layer
}

-- Create the neural network
local network = luann:new(config)

-- Dummy dataset
local inputs = {
    {0.1, 0.2, 0.7},
    {0.3, 0.4, 0.3},
    {0.6, 0.2, 0.2}
}

local targets = {
    {1, 0},  -- First class
    {0, 1},  -- Second class
    {1, 0}   -- First class
}

-- Activation functions for each layer
local activationFuncs = {
    {"leakyRelu", 0.01},  -- LeakyReLU for hidden layer
    {"softmax"}           -- Softmax for output layer
}

-- Training loop
for epoch = 1, 1000 do
    for i, input in ipairs(inputs) do
        network:backpropagate(input, targets[i], activationFuncs, {0.9, 0.999})
    end
end

-- Testing the network
for _, input in ipairs(inputs) do
    local output = network:activate(input, activationFuncs, false)
    print(table.concat(output, ", "))
end
