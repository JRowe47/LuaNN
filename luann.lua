local luann = {}
local Layer = {}
local Cell = {}
local exp = math.exp

-- Weight initialization methods
local weightInits = {
    default = function(numInputs)
        return math.random() * 0.1
    end,
    xavier = function(numInputs, numOutputs)
        local stdv = math.sqrt(2 / (numInputs + numOutputs))
        return math.random() * stdv
    end
}

-- Cell object for individual neurons
function Cell:new(numInputs, weightInitMethod, numOutputs)
    local cell = {delta = 0, weights = {}, signal = 0}
    local initMethod = weightInits[weightInitMethod] or weightInits.default
    for i = 1, numInputs do
        cell.weights[i] = initMethod(numInputs, numOutputs)
    end
    setmetatable(cell, self)
    self.__index = self
    return cell
end

function Cell:weightedSum(inputs, bias)
    local sum = bias or 0
    for i = 1, #self.weights do
        sum = sum + (self.weights[i] * inputs[i])
    end
    return sum
end

function Layer:new(numCells, numInputs, weightInitMethod)
    local cells, biases = {}, {}
    for i = 1, numCells do
        cells[i] = Cell:new(numInputs, weightInitMethod, numCells)
        biases[i] = math.random() * 0.1
    end
    local layer = {cells = cells, biases = biases}
    setmetatable(layer, self)
    self.__index = self
    return layer
end

-- Activation functions
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
    local max = math.max(table.unpack(self:getSignals()))
    local exps = {}
    local sumExps = 0
    for i, cell in ipairs(self.cells) do
        exps[i] = exp(cell.signal - max)
        sumExps = sumExps + exps[i]
    end
    for i, cell in ipairs(self.cells) do
        cell.signal = exps[i] / sumExps
    end
end

-- SoftMax derivative (also layer-wide)
function Layer:softmaxDerivative(outputIndex)
    local derivatives = {}
    for i, cell in ipairs(self.cells) do
        derivatives[i] = i == outputIndex and cell.signal * (1 - cell.signal) or -cell.signal * self.cells[outputIndex].signal
    end
    return derivatives
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