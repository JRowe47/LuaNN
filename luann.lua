--[[
    LuaNN: A Neural Network Library for Lua and LuaJIT

    Overview:
    LuaNN provides a flexible framework for building and training neural networks in Lua and LuaJIT environments. It features a simple yet powerful API for creating various types of neural network architectures, including support for custom layers, activation functions, and optimization algorithms.

    Features:
    - Layer abstraction: Define neural network layers with customizable neuron count, weight initialization, and dropout.
    - Activation functions: Includes common functions like sigmoid, ReLU, and their derivatives.
    - Weight Initialization: Support for default and Xavier initialization methods.
    - AdamW Optimization: Incorporates AdamW optimizer for weight updates, including momentum and velocity terms.
    - Dropout: Implements dropout functionality for regularization during training.
    - Softmax: Layer-wide softmax function and its derivative for probability distribution outputs.
    - Backpropagation: Supports backpropagation with L1 and L2 regularization, and batch backpropagation for training.
    - Forward Forward learning integration
    
    Usage:
    1. Initialize the network with desired layer sizes, learning rate, weight initialization method, and regularization parameters.
    2. Propagate signals through the network using specified activation functions.
    3. Apply backpropagation to train the network with given inputs and target outputs.

    Example:
    ```
    local nn = luann:new({64, 128, 64, 10}, 0.001, 'xavier', 0.01, 0.01, {0.2, 0.5, 0.2, 0})
    local inputs = {0.5, 0.1, -0.3, ...}  -- Example input array
    local outputs = nn:activate(inputs, {'relu', 'relu', 'sigmoid', 'softmax'})
    ```

    Note:
    This library is designed for use in Lua and LuaJIT environments. Performance optimizations specific to LuaJIT may be applied for enhanced efficiency in computationally intensive operations.

    Author: JRowe47
    Version: 1.1
    License: MIT
]]

local luann = {}
local Layer = {}
local cell = {}

local exponential = math.exp
local random = math.random

local activations = {}

    activations.sigmoid = function(x) return 1 / (1 + exponential(-x)) end
    activations.sigmoid_derivative = function(x) local s = activations.sigmoid(x); return s * (1 - s) end
    
    activations.relu = function(x) return math.max(0, x) end
    activations.relu_derivative = function(x) return x > 0 and 1 or 0 end
    
    activations.tanh = function(x) return math.tanh(x) end
    activations.tanh_derivative = function(x) local t = math.tanh(x); return 1 - t * t end
    
    activations.leakyRelu = function(x, alpha) return x > 0 and x or alpha * x end
    activations.leakyRelu_derivative = function(x, alpha) return x > 0 and 1 or alpha end
    
    activations.elu = function(x, alpha) return x > 0 and x or alpha * (exponential(x) - 1) end
    activations.elu_derivative = function(x, alpha) local e = exponential(x); return x > 0 and 1 or alpha * e end
    
    
    activations.selu = function(x, lambda, alpha) return x > 0 and lambda * x or lambda * alpha * (exponential(x) - 1) end
    activations.selu_derivative = function(x, lambda, alpha) local e = exponential(x); return x > 0 and lambda or lambda * alpha * e end
    
    activations.swish = function(x, beta) return x * (1 / (1 + exponential(-beta * x))) end
    activations.swish_derivative = function(x, beta) local sig = 1 / (1 + exp(-beta * x)); return sig + beta * x * sig * (1 - sig) end
    
    -- Add softmax and its derivative to the activations table
    activations.softmax = function(x)
        local max = -math.huge
        for i = 1, #x do
            if x[i] > max then max = x[i] end
        end
        local sum = 0
        local result = {}
        for i = 1, #x do
            result[i] = math.exp(x[i] - max)
            sum = sum + result[i]
        end
        for i = 1, #x do
            result[i] = result[i] / sum
        end
        return result
    end

-- Weight initialization strategies
local weightInits = {
    default = function(numInputs)
        return random() * 0.1
    end,
    xavier = function(numInputs, numOutputs)
        local stdv = math.sqrt(2 /
          (numInputs + numOutputs))
        return random() * 2 * stdv - stdv
    end
}

function cell:new(numInputs, weightInitMethod, numOutputs)
    local cellInstance = {
      delta = 0,
      weights = {},
      signal = 0,
      m = {bias = 0},  -- Initialize momentum for bias
      v = {bias = 0},  -- Initialize velocity for bias
      timestep = 0
    }
    local initMethod = weightInits[weightInitMethod] or weightInits.default
    for i = 1, numInputs do
        cellInstance.m[i] = 0
        cellInstance.v[i] = 0
        cellInstance.weights[i] = initMethod(numInputs, numOutputs)
    end
    setmetatable(cellInstance, { __index = cell })
    return cellInstance
end

function cell:weightedSum(inputs, bias)
    local sum = bias or 0
    for i = 1, #self.weights do
        sum = sum + self.weights[i] * inputs[i]
    end
    return sum
end

function Layer:new(numCells, numInputs, weightInitMethod, dropoutRate)
    local layerInstance = {
        cells = {},
        biases = {},
        dropoutRate = dropoutRate or 0
    }
    for i = 1, numCells do
        layerInstance.cells[i] = cell:new(numInputs, weightInitMethod, numCells)
        layerInstance.biases[i] = random() * 0.1
    end
    setmetatable(layerInstance, { __index = Layer })
    return layerInstance
end

function Layer:applyDropout(isTraining)
    if isTraining and self.dropoutRate > 0 then
        for _, cell in ipairs(self.cells) do
            if math.random() < self.dropoutRate then
                cell.signal = 0  -- Dropping out the neuron
            end
        end
    end
end

function luann:new(configuration)
    local networkInstance = {
        learningRate = configuration.learningRate,
        layers = {},
        l1Lambda = configuration.l1Lambda or 0,
        l2Lambda = configuration.l2Lambda or 0
    }
    for i, layerConfig in ipairs(configuration.layers) do
        local numInputs
        local numCells
        local weightInitMethod
        local dropoutRate = configuration.dropoutRates and configuration.dropoutRates[i] or 0

        if i == 1 then
            -- First layer (input layer), layerConfig is just a number
            numInputs = layerConfig
            numCells = layerConfig
            weightInitMethod = configuration.weightInitMethod
        else
            -- Subsequent layers, layerConfig is a table
            numInputs = #networkInstance.layers[i - 1].cells
            numCells = layerConfig.numCells
            weightInitMethod = layerConfig.weightInitMethod or configuration.weightInitMethod
        end

        networkInstance.layers[i] = Layer:new(numCells, numInputs, weightInitMethod, dropoutRate)
    end
    setmetatable(networkInstance, { __index = luann })
    return networkInstance
end

function luann:setInputSignals(inputs)
    for i = 1, #inputs do
        self.layers[1].cells[i].signal = inputs[i]
    end
end

function luann:updateLearningRate(newLearningRate)
    self.learningRate = newLearningRate
end

function luann:propagateSignals(activationFuncs)
    for i = 2, #self.layers do
        local layer, prevLayer = self.layers[i], self.layers[i-1]
        local activationFuncInfo = activationFuncs[i - 1]
        local activationFunc = activations[activationFuncInfo[1]]
        if not activationFunc then
            error("Activation function '" .. tostring(activationFuncInfo[1]) .. "' does not exist")
        end

        if activationFuncInfo[1] == "softmax" and i == #self.layers then
            -- Applying softmax on the final layer
            local inputs = {}
            for j, cell in ipairs(layer.cells) do
                inputs[j] = cell:weightedSum(self:getSignals(prevLayer), layer.biases[j])
            end
            local softmaxOutputs = activationFunc(inputs)
            for j, cell in ipairs(layer.cells) do
                cell.signal = softmaxOutputs[j]
            end
        else
            for j, cell in ipairs(layer.cells) do
                local sum = cell:weightedSum(self:getSignals(prevLayer), layer.biases[j])
                cell.signal = activationFunc(sum, table.unpack(activationFuncInfo, 2))
            end
        end
    end
end

function luann:propagateSignalsFF(activationFuncs, data, isPositive)
    local goodness = {}  -- Stores the goodness for each layer

    for i = 2, #self.layers do
        local layer = self.layers[i]
        local prevLayer = self.layers[i-1]
        local activationFuncInfo = activationFuncs[i - 1]
        local activationFunc = activations[activationFuncInfo[1]]
        
        if not activationFunc then
            error("Activation function '" .. tostring(activationFuncInfo[1]) .. "' does not exist")
        end

        -- Calculate the weighted sum and apply activation function
        for j, cell in ipairs(layer.cells) do
            local sum = cell:weightedSum(self:getSignals(prevLayer), layer.biases[j])
            cell.signal = activationFunc(sum, table.unpack(activationFuncInfo, 2))

            -- Layer normalization (if needed)
            if i < #self.layers then  -- Exclude output layer from normalization
                cell.signal = self:layerNormalize(cell.signal, prevLayer)
            end
        end

        -- Calculate the goodness for this layer
        goodness[i] = self:calculateGoodness(self:getSignals(layer))

        -- Adjust weights for Forward-Forward learning
        if isPositive then
            self:adjustWeights(layer, data[i], true)  -- Positive adjustment
        else
            self:adjustWeights(layer, data[i], false)  -- Negative adjustment
        end
    end

    return goodness
end

-- Function to calculate the goodness of a layer
function luann:calculateGoodness(signals)
    local goodness = 0
    for _, signal in ipairs(signals) do
        goodness = goodness + signal ^ 2
    end
    return goodness
end

-- Function to perform layer normalization
function luann:layerNormalize(signal, layer)
    local sum = 0
    for _, cell in ipairs(layer.cells) do
        sum = sum + cell.signal ^ 2
    end
    local normFactor = math.sqrt(sum)
    return signal / normFactor
end

-- Function to adjust weights for FF learning
function luann:adjustWeights(layer, data, isPositive)
    local adjustmentFactor = isPositive and 1 or -1
    local learningRate = self.learningRate * adjustmentFactor

    for _, cell in ipairs(layer.cells) do
        for i = 1, #cell.weights do
            local weightUpdate = learningRate * data[i] * cell.delta
            cell.weights[i] = cell.weights[i] + weightUpdate
        end
    end
end

function luann:getSignals(layer)
    local signals = {}
    for _, cell in ipairs(layer.cells) do
        table.insert(signals, cell.signal)
    end
    return signals
end

function luann:activate(inputs, activationFuncs, isTraining)
    self:setInputSignals(inputs)
    self:propagateSignals(activationFuncs)
    if isTraining then
        for _, layer in ipairs(self.layers) do
            layer:applyDropout(isTraining)
        end
    end
    return self:getSignals(self.layers[#self.layers])
end

function luann:backpropagate(inputs, targetOutputs, activationFuncs, adamParams)
    local beta1 = adamParams[1] or 0.9
    local beta2 = adamParams[2] or 0.999
    local epsilon = adamParams.epsilon or 1e-8
    local weightDecay = adamParams.weightDecay or 0

    -- Activate the network with the current inputs and activation functions
    self:activate(inputs, activationFuncs, false)  -- Assuming isTraining is false during backpropagation

    for i = #self.layers, 2, -1 do
        local layer = self.layers[i]
        local prevLayerSignals = self:getSignals(self.layers[i-1])
        local activationFuncName = activationFuncs[i - 1][1]
        local derivativeFunc = activations[activationFuncName .. '_derivative']

        for j, cell in ipairs(layer.cells) do
            cell.timestep = cell.timestep + 1
            local errorTerm = 0

            if i == #self.layers then
                -- Special handling for the output layer with softmax activation
                if activationFuncName == "softmax" then
                    -- Simplified derivative for softmax with cross-entropy loss
                    errorTerm = cell.signal - targetOutputs[j]
                else
                    -- For other activation functions in the output layer
                    if not derivativeFunc then
                        error("Derivative function for '" .. activationFuncName .. "' does not exist")
                    end
                    errorTerm = (cell.signal - targetOutputs[j]) * derivativeFunc(cell.signal)
                end
            else
                -- Hidden layers
                if not derivativeFunc then
                    error("Derivative function for '" .. activationFuncName .. "' does not exist")
                end
                for k, nextcell in ipairs(self.layers[i+1].cells) do
                    errorTerm = errorTerm + nextcell.delta * nextcell.weights[j]
                end
                errorTerm = errorTerm * derivativeFunc(cell.signal, table.unpack(activationFuncs[i - 1], 2))
            end

            cell.delta = errorTerm

            -- Update weights and biases using Adam optimization
            for k, inputSignal in ipairs(prevLayerSignals) do
                local grad = cell.delta * inputSignal
                cell.m[k] = beta1 * cell.m[k] + (1 - beta1) * grad
                cell.v[k] = beta2 * cell.v[k] + (1 - beta2) * grad * grad
                local mHat = cell.m[k] / (1 - beta1 ^ cell.timestep)
                local vHat = cell.v[k] / (1 - beta2 ^ cell.timestep)
                cell.weights[k] = cell.weights[k] * (1 - weightDecay) - (self.learningRate * mHat) / (math.sqrt(vHat) + epsilon)
            end

            local biasGrad = cell.delta
            cell.m.bias = beta1 * cell.m.bias + (1 - beta1) * biasGrad
            cell.v.bias = beta2 * cell.v.bias + (1 - beta2) * biasGrad * biasGrad
            local mHatBias = cell.m.bias / (1 - beta1 ^ cell.timestep)
            local vHatBias = cell.v.bias / (1 - beta2 ^ cell.timestep)
            layer.biases[j] = layer.biases[j] * (1 - weightDecay) - (self.learningRate * mHatBias) / (math.sqrt(vHatBias) + epsilon)
        end
    end
end



return luann