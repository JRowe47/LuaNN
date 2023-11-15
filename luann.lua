--[[
LuaJIT Neural Network Library (LuaNN)

Overview:
LuaNN provides a lightweight and efficient neural network implementation in LuaJIT. It's designed for simplicity and speed, leveraging LuaJIT's JIT compilation and FFI capabilities.


Features:
- Efficient JIT Compilation: LuaNN is optimized for LuaJIT's JIT compiler, ensuring fast execution.
- Weight Initialization Methods: Supports various methods including 'default', 'Xavier', and 'He' initialization for enhanced network stability.
- Neuron Representation: Employs a detailed representation of neurons, including weighted inputs, biases, and activation functions.
- Layer Abstraction: Facilitates easy construction of neural networks by managing collections of neurons in layers.
- Activation Functions: Includes sigmoid, ReLU, tanh, LeakyReLU, ELU, and more, vital for non-linear processing in networks.
- Derivatives for Backpropagation: Implements derivatives of various activation functions, vital for effective training using backpropagation.
- SoftMax Functionality: Provides SoftMax and its derivative for categorization tasks in output layers.
- Advanced Learning Algorithms: Supports various optimization algorithms like AdamW, learning rate scheduling, etc., for efficient training.
- Attention Mechanism: Includes attention layers to enhance learning from complex data structures, with backpropagation support for these layers.
- Modular Design: LuaNN's modular structure allows for easy customization and expansion.

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

This example outlines learning rate scheduling:
    -- Adjust learning rate based on your scheduling strategy
    if epoch % 10 == 0 then  -- Example: Reduce learning rate every 10 epochs
        network:updateLearningRate(network.learningRate * 0.9)
    end

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
    local cell = {
      delta = 0, 
      weights = {}, 
      signal = 0,
      m = {},  -- Momentum for AdamW
      v = {},  -- Velocity for AdamW
      timestep = 0  -- Timestep for AdamW
      }
    local initMethod = weightInits[weightInitMethod] or weightInits.default
    -- Initialize each weight using the selected weight initialization method
    for i = 1, numInputs do
        cell.m[i] = 0
        cell.v[i] = 0
    end
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
function Layer:new(numCells, numInputs, weightInitMethod, dropoutRate)
    local cells, biases = {}, {}
    self.dropoutRate = dropoutRate or 0  -- Default dropout rate is 0 (no dropout)
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

-- Layer dropout functionality
function Layer:applyDropout(isTraining)
    if isTraining and self.dropoutRate > 0 then
        for _, cell in ipairs(self.cells) do
            if math.random() < self.dropoutRate then
                cell.signal = 0  -- Dropping out the neuron
            end
        end
    end
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

-- Modify the Layer class to handle batches of inputs
function Layer:forwardBatch(batchInputs)
    -- batchInputs is a matrix where each row is an input instance
    -- Initialize a matrix to store the outputs for each input instance
    local batchOutputs = {}
    for i = 1, #batchInputs do
        local outputs = {}
        for j, cell in ipairs(self.cells) do
            outputs[j] = cell:weightedSum(batchInputs[i], self.biases[j])
        end
        batchOutputs[i] = outputs
    end
    return batchOutputs
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


-- Initialize the neural network with given layer sizes, learning rate, weight initialization method, and regularization parameters
function luann:new(layers, learningRate, weightInitMethod, l1Lambda, l2Lambda, dropoutRates)
    local network = {learningRate = learningRate, layers = {}, l1Lambda = l1Lambda or 0, l2Lambda = l2Lambda or 0}
    network.layers[1] = Layer:new(layers[1], layers[1], weightInitMethod, dropoutRates[1])
    for i = 2, #layers do
        network.layers[i] = Layer:new(layers[i], layers[i-1], weightInitMethod, dropoutRates[i])
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

-- Function to update the learning rate
function luann:updateLearningRate(newLearningRate)
    self.learningRate = newLearningRate
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
function luann:activate(inputs, activationFuncs, isTraining)
    self:setInputSignals(inputs)
    self:propagateSignals(activationFuncs)
    -- Apply dropout if in training mode
    if isTraining then
        for _, layer in ipairs(self.layers) do
            layer:applyDropout(isTraining)
        end
    end
    return self:getSignals(self.layers[#self.layers])
end

-- Modify the luann class to handle batches of inputs
function luann:activateBatch(batchInputs, activationFuncs, isTraining)
    local currentBatch = batchInputs
    for i = 1, #self.layers do
        local layerOutputs = self.layers[i]:forwardBatch(currentBatch)
        -- Apply activation functions to each output in the batch
        for j = 1, #layerOutputs do
            local activationFunc = activations[activationFuncs[i][1]]
            for k = 1, #layerOutputs[j] do
                layerOutputs[j][k] = activationFunc(layerOutputs[j][k])
            end
        end
        currentBatch = layerOutputs
        -- Apply dropout if in training mode
        if isTraining then
            self.layers[i]:applyDropout(isTraining)
        end
    end
    return currentBatch
end

-- Backpropagation training algorithm with L1 and L2 regularization
function luann:backpropagate(inputs, targetOutputs, activationFuncs, adamParams)
    -- AdamW parameters
    local beta1 = adamParams.beta1 or 0.9
    local beta2 = adamParams.beta2 or 0.999
    local epsilon = adamParams.epsilon or 1e-8
    local weightDecay = adamParams.weightDecay or 0

    self:activate(inputs, activationFuncs)

    for i = #self.layers, 2, -1 do
        local layer = self.layers[i]
        local prevLayerSignals = self:getSignals(self.layers[i-1])
        local activationFuncName = activationFuncs[i - 1][1]
        local isSoftMaxLayer = activationFuncName == 'softmax'

        for j, cell in ipairs(layer.cells) do
            cell.timestep = cell.timestep + 1

            local errorTerm
            if i == #self.layers then
                errorTerm = isSoftMaxLayer and (cell.signal - targetOutputs[j]) or (targetOutputs[j] - cell.signal)
            else
                errorTerm = 0
                for k, nextCell in ipairs(self.layers[i+1].cells) do
                    errorTerm = errorTerm + nextCell.delta * nextCell.weights[j]
                end
            end

            local derivativeFunc = isSoftMaxLayer and function(x) return x end or activationDerivatives[activationFuncName]
            if not derivativeFunc then
                error("Derivative function for '" .. tostring(activationFuncName) .. "' does not exist")
            end
            cell.delta = errorTerm * derivativeFunc(cell.signal, table.unpack(activationFuncs[i - 1], 2))

            -- AdamW weight and bias updates
            for k, inputSignal in ipairs(prevLayerSignals) do
                local grad = cell.delta * inputSignal

                -- Momentum and velocity updates
                cell.m[k] = beta1 * cell.m[k] + (1 - beta1) * grad
                cell.v[k] = beta2 * cell.v[k] + (1 - beta2) * grad * grad

                -- Bias-corrected estimates
                local mHat = cell.m[k] / (1 - beta1 ^ cell.timestep)
                local vHat = cell.v[k] / (1 - beta2 ^ cell.timestep)

                -- L1 and L2 regularization
                local l1Reg = self.l1Lambda * (cell.weights[k] > 0 and 1 or -1)
                local l2Reg = self.l2Lambda * cell.weights[k]

                -- Weight update with AdamW, L1, and L2
                cell.weights[k] = cell.weights[k] * (1 - weightDecay) - (self.learningRate * mHat) / (math.sqrt(vHat) + epsilon) - (self.learningRate * l1Reg) - (self.learningRate * l2Reg)
            end

            -- Bias update (if you have separate biases)
            local biasGrad = cell.delta
            cell.m.bias = beta1 * cell.m.bias + (1 - beta1) * biasGrad
            cell.v.bias = beta2 * cell.v.bias + (1 - beta2) * biasGrad * biasGrad

            local mHatBias = cell.m.bias / (1 - beta1 ^ cell.timestep)
            local vHatBias = cell.v.bias / (1 - beta2 ^ cell.timestep)

            layer.biases[j] = layer.biases[j] * (1 - weightDecay) - (self.learningRate * mHatBias) / (math.sqrt(vHatBias) + epsilon)
        end
    end
end

function luann:batchBackpropagate(batchInputs, batchTargetOutputs, activationFuncs, adamParams)
    local batchSize = #batchInputs

    -- Initialize gradients accumulation for each layer
    local layerGradients = {}
    for i, layer in ipairs(self.layers) do
        layerGradients[i] = {
            weights = {},
            biases = {}
        }
        for j, cell in ipairs(layer.cells) do
            layerGradients[i].weights[j] = {}
            for k = 1, #cell.weights do
                layerGradients[i].weights[j][k] = 0
            end
            layerGradients[i].biases[j] = 0
        end
    end

    -- Process each instance in the batch
    for b = 1, batchSize do
        self:activate(batchInputs[b], activationFuncs)  -- Forward pass

        -- Backward pass for each layer
        for i = #self.layers, 1, -1 do
            local layer = self.layers[i]
            local prevLayerSignals = self:getSignals(self.layers[i-1])
            local activationFuncName = activationFuncs[i][1]
            local isSoftMaxLayer = activationFuncName == 'softmax'

            for j, cell in ipairs(layer.cells) do
                -- Compute error term (delta)
                local errorTerm
                if i == #self.layers then
                    errorTerm = isSoftMaxLayer and (cell.signal - batchTargetOutputs[b][j]) or (batchTargetOutputs[b][j] - cell.signal)
                else
                    errorTerm = 0
                    for k, nextCell in ipairs(self.layers[i+1].cells) do
                        errorTerm = errorTerm + nextCell.delta * nextCell.weights[j]
                    end
                end

                local derivativeFunc = isSoftMaxLayer and function(x) return x end or activationDerivatives[activationFuncName]
                cell.delta = errorTerm * derivativeFunc(cell.signal)

                -- Accumulate gradients for weights and biases
                for k, inputSignal in ipairs(prevLayerSignals) do
                    layerGradients[i].weights[j][k] = layerGradients[i].weights[j][k] + cell.delta * inputSignal
                end
                layerGradients[i].biases[j] = layerGradients[i].biases[j] + cell.delta
            end
        end
    end

    -- Update weights and biases using the average gradients
    for i, layer in ipairs(self.layers) do
        for j, cell in ipairs(layer.cells) do
            for k, weight in ipairs(cell.weights) do
                cell.weights[k] = weight - self.learningRate * (layerGradients[i].weights[j][k] / batchSize)
            end
            layer.biases[j] = layer.biases[j] - self.learningRate * (layerGradients[i].biases[j] / batchSize)
        end
    end
end

return luann