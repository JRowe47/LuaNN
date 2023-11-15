--[[
LuaJIT Neural Network Library (LuaNN)

Overview:
LuaNN provides a lightweight and efficient neural network implementation in LuaJIT. It's designed for simplicity and speed, leveraging LuaJIT's JIT compilation and FFI capabilities.

Features:
- Weight Initialization Methods: Includes 'default' random initialization and 'Xavier' initialization for improved training stability.
- Neuron Representation (Cell): Models individual neurons, handling weighted input sum calculation.
- Layer Abstraction: Manages a collection of neurons, simplifying network construction.
- Activation Functions: Supports common functions like sigmoid, ReLU, and tanh, essential for non-linear transformations in neural networks.
- Derivatives for Backpropagation: Implements derivatives of activation functions, crucial for the backpropagation training algorithm.
- SoftMax Functionality: Provides SoftMax and its derivative for output layers, useful in classification tasks.
- Attention Mechanism: Implements attention layers that can focus on specific parts of the input, enhancing the model's ability to learn from complex data structures.
- Attention Backpropagation: Supports the backpropagation process through attention layers, optimizing attention weights based on training data.
- Neural Network Construction (luann): Facilitates creating, training, and using neural networks, encapsulating the entire workflow.

Example Usage:
1. Network Creation:
   -- Initialize a neural network with 784 input neurons, 100 neurons in the hidden layer, and 10 output neurons.
   -- The learning rate is set to 0.01 and the Xavier weight initialization method is used.
   local myNetwork = luann:new({784, 100, 10}, 0.01, 'xavier')

2. Input Setting:
   -- Assuming 'inputData' is a table representing your input data (e.g., image pixel values).
   myNetwork:setInputSignals(inputData)

3. Activation Function Assignment:
   -- Assign 'relu' for the hidden layer and 'softmax' for the output layer.
   local activationFunctions = {
       { 'relu' },     -- First hidden layer
       { 'softmax' }   -- Output layer
   }

4. Training the Network:
   -- 'inputData' is the input data, and 'targetOutputs' are the desired outputs for training.
   myNetwork:backpropagate(inputData, targetOutputs, activationFunctions)

5. Making Predictions:
   -- Pass new input data to the network to get predictions.
   local predictions = myNetwork:activate(newInputData, activationFunctions)

Note: 'inputData', 'targetOutputs', and 'newInputData' are placeholders.
This example outlines the basic steps for network creation, training, and prediction.

--]]

local luann = {}
local Layer = {}
local Cell = {}
local exp = math.exp

-- Weight initialization methods
local weightInits = {
    -- Default initialization method
    default = function(numInputs)
        return math.random() * 0.1
    end,
    -- Xavier initialization method for symmetry breaking
    xavier = function(numInputs, numOutputs)
        local stdv = math.sqrt(2 / (numInputs + numOutputs))
        return math.random() * 2 * stdv - stdv  -- Generates weights in the range [-stdv, stdv]
    end
}

-- Cell object for individual neurons
function Cell:new(numInputs, weightInitMethod, numOutputs)
    -- Initialize cell with weights, delta (error term), and signal (output)
    local cell = {delta = 0, weights = {}, signal = 0}
    local initMethod = weightInits[weightInitMethod] or weightInits.default
    -- Initialize each weight using the selected weight initialization method
    for i = 1, numInputs do
        cell.weights[i] = initMethod(numInputs, numOutputs)
    end
    setmetatable(cell, self)
    self.__index = self
    return cell
end

-- Calculate the weighted sum of inputs for a neuron
function Cell:weightedSum(inputs, bias)
    local sum = bias or 0
    -- Sum the product of each input with its corresponding weight
    for i = 1, #self.weights do
        sum = sum + (self.weights[i] * inputs[i])
    end
    return sum
end

-- Layer object, representing a layer of cells (neurons)
function Layer:new(numCells, numInputs, weightInitMethod)
    local cells, biases = {}, {}
    -- Initialize each cell and bias in the layer
    for i = 1, numCells do
        cells[i] = Cell:new(numInputs, weightInitMethod, numCells)
        biases[i] = math.random() * 0.1  -- Random bias initialization
    end
    local layer = {cells = cells, biases = biases}
    setmetatable(layer, self)
    self.__index = self
    return layer
end

-- Attention layer
function Layer:attention(inputs, attentionSize)
    local attentionWeights = {}
    local attentionBiases = {}
    local outputs = {}

    -- Initialize weights and biases for attention
    for i = 1, attentionSize do
        attentionWeights[i] = math.random() * 0.1 -- Random initialization, can be improved
        attentionBiases[i] = math.random() * 0.1
    end

    -- Compute attention scores and apply them to inputs
    for i, cell in ipairs(self.cells) do
        local attentionScore = 0
        for j = 1, attentionSize do
            attentionScore = attentionScore + (attentionWeights[j] * inputs[j]) + attentionBiases[j]
        end

        -- Apply attention score (e.g., via softmax)
        local weightedInput = exp(attentionScore) * cell.signal
        table.insert(outputs, weightedInput)
    end

    -- Normalize outputs (you can use softmax or another normalization method)
    local sumOutputs = 0
    for _, output in ipairs(outputs) do
        sumOutputs = sumOutputs + output
    end
    for i = 1, #outputs do
        outputs[i] = outputs[i] / sumOutputs
    end

    return outputs
end


-- Activation functions
  -- sigmoid, relu, tan, leakyRelu, elu, selu, and swish
local activations = {
    sigmoid = function(x) return 1 / (1 + exp(-x)) end,
    relu = function(x) return math.max(0, x) end,
    tanh = function(x) return math.tanh(x) end,
    leakyRelu = function(x, alpha) return x > 0 and x or alpha * x end,
    elu = function(x, alpha) return x > 0 and x or alpha * (exp(x) - 1) end,
    selu = function(x, lambda, alpha) return x > 0 and lambda * x or lambda * alpha * (exp(x) - 1) end,
    swish = function(x, beta) return x * (1 / (1 + exp(-beta * x))) end
}

-- Derivatives of activation functions
  -- sigmoid, relu, tan, leakyRelu, elu, selu, and swish
local activationDerivatives = {
    sigmoid = function(x) local s = activations.sigmoid(x); return s * (1 - s) end,
    relu = function(x) return x > 0 and 1 or 0 end,
    tanh = function(x) local t = math.tanh(x); return 1 - t * t end,
    leakyRelu = function(x, alpha) return x > 0 and 1 or alpha end,
    elu = function(x, alpha) local e = exp(x); return x > 0 and 1 or alpha * e end,
    selu = function(x, lambda, alpha) local e = exp(x); return x > 0 and lambda or lambda * alpha * e end,
    swish = function(x, beta) local sig = 1 / (1 + exp(-beta * x)); return sig + beta * x * sig * (1 - sig) end
}

-- SoftMax implementation (layer-wide)
function Layer:softmax()
    -- Softmax function to convert neuron signals into probabilities
    local max = math.max(table.unpack(self:getSignals()))
    local exps = {}
    local sumExps = 0
    -- Calculating exponentials and their sum for normalization
    for i, cell in ipairs(self.cells) do
        exps[i] = exp(cell.signal - max)  -- Stability improvement by subtracting max
        sumExps = sumExps + exps[i]
    end
    -- Normalizing each cell's signal to get probabilities
    for i, cell in ipairs(self.cells) do
        cell.signal = exps[i] / sumExps
    end
end

-- SoftMax derivative (also layer-wide)
function Layer:softmaxDerivative(outputIndex)
    local derivatives = {}
    -- Calculating the derivative for each cell's signal
    for i, cell in ipairs(self.cells) do
        derivatives[i] = i == outputIndex and cell.signal * (1 - cell.signal) or -cell.signal * self.cells[outputIndex].signal
    end
    return derivatives
end

function Layer:attentionBackPropagation(inputs, targetOutputs, attentionSize, learningRate)
    local attentionWeightsGradient = {}
    local attentionBiasesGradient = {}
    
    -- Initialize gradients to zero
    for i = 1, attentionSize do
        attentionWeightsGradient[i] = 0
        attentionBiasesGradient[i] = 0
    end

    -- Compute gradients for attention weights and biases
    for i, cell in ipairs(self.cells) do
        local derivative = targetOutputs[i] - cell.signal
        for j = 1, attentionSize do
            attentionWeightsGradient[j] = attentionWeightsGradient[j] + derivative * inputs[j]
            attentionBiasesGradient[j] = attentionBiasesGradient[j] + derivative
        end
    end

    -- Update attention weights and biases
    for i = 1, attentionSize do
        self.attentionWeights[i] = self.attentionWeights[i] - learningRate * attentionWeightsGradient[i]
        self.attentionBiases[i] = self.attentionBiases[i] - learningRate * attentionBiasesGradient[i]
    end
end


-- Initialize the neural network with given layer sizes, learning rate, and weight initialization method
function luann:new(layers, learningRate, weightInitMethod)
    local network = {learningRate = learningRate, layers = {}}
    network.layers[1] = Layer:new(layers[1], layers[1], weightInitMethod)
    for i = 2, #layers do
        network.layers[i] = Layer:new(layers[i], layers[i-1], weightInitMethod)
    end
    setmetatable(network, self)
    self.__index = self
    return network
end

function luann:setInputSignals(inputs)
    for i = 1, #inputs do
        self.layers[1].cells[i].signal = inputs[i]
    end
end

-- Forward propagation of signals through the network
function luann:propagateSignals(activationFuncs)
    for i = 2, #self.layers do
        local layer, prevLayer = self.layers[i], self.layers[i-1]
        local activationFuncInfo = activationFuncs[i - 1]

        local activationFunc = activations[activationFuncInfo[1]]
        if not activationFunc then
            error("Activation function '" .. tostring(activationFuncInfo[1]) .. "' does not exist")
        end

        for j, cell in ipairs(layer.cells) do
            local sum = cell:weightedSum(self:getSignals(prevLayer), layer.biases[j])
            -- Passing all parameters after the first one from activationFuncInfo
            cell.signal = activationFunc(sum, table.unpack(activationFuncInfo, 2))
        end
    end
    if activationFuncs[#activationFuncs][1] == 'softmax' then
        self.layers[#self.layers]:softmax()
    end
end

-- Get signals from a layer
function luann:getSignals(layer)
    local signals = {}
    for _, cell in ipairs(layer.cells) do
        table.insert(signals, cell.signal)
    end
    return signals
end

-- Main activation function (refactored)
function luann:activate(inputs, activationFuncs)
    self:setInputSignals(inputs)
    self:propagateSignals(activationFuncs)
    return self:getSignals(self.layers[#self.layers])
end

-- Backpropagation training algorithm
function luann:backpropagate(inputs, targetOutputs, activationFuncs)
    self:activate(inputs, activationFuncs)

    for i = #self.layers, 2, -1 do
        local layer = self.layers[i]
        local prevLayerSignals = self:getSignals(self.layers[i-1])
        local activationFuncName = activationFuncs[i - 1][1]
        -- Check if current layer uses SoftMax
        local isSoftMaxLayer = activationFuncs[i - 1] == 'softmax'

        for j, cell in ipairs(layer.cells) do
            local errorTerm
            if i == #self.layers then
                if isSoftMaxLayer then
                    -- SoftMax-specific error calculation
                    errorTerm = cell.signal - targetOutputs[j]
                else
                    -- Standard error calculation for other activation functions
                    errorTerm = targetOutputs[j] - cell.signal
                end
            else
                errorTerm = 0
                for k, nextCell in ipairs(self.layers[i+1].cells) do
                    errorTerm = errorTerm + nextCell.delta * nextCell.weights[j]
                end
            end

            local derivative
            if isSoftMaxLayer then
                -- SoftMax derivative is simply the signal for the correct class minus 1
                derivative = (j == targetOutputs and cell.signal - 1) or cell.signal
            else
                -- Derivative for other activation functions
                local derivativeFunc = activationDerivatives[activationFuncName]
                if not derivativeFunc then
                    error("Derivative function for '" .. tostring(activationFuncName) .. "' does not exist")
                end
                derivative = derivativeFunc(cell.signal, table.unpack(activationFuncs[i - 1], 2))
                --derivative = derivativeFunc(cell.signal)
            end

            cell.delta = errorTerm * derivative

            for k, inputSignal in ipairs(prevLayerSignals) do
                cell.weights[k] = cell.weights[k] + self.learningRate * cell.delta * inputSignal
            end
            layer.biases[j] = layer.biases[j] + self.learningRate * cell.delta
        end
    end
end

return luann