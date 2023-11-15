-- Require the luann library
local luann = require 'luannUpdated'

-- Seed the random number generator
math.randomseed(os.time())

-- Function to print table (for displaying outputs)
local function printTable(t)
    for i, v in ipairs(t) do
        io.write(string.format("%.5f ", v))
    end
    io.write("\n")
end

-- XOR input and output pairs
local xor_inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}
local xor_outputs = {{0}, {1}, {1}, {0}}

-- Activation functions for each layer (assuming two layers)
local activationFuncs = {
    {"swish", 0.5},  -- for the first hidden layer
    {"swish", 0.5}   -- for the output layer
}

-- Create the XOR network (2 inputs, 2 neurons in hidden layer, 1 output)
local xor_net = luann:new({2, 2, 1}, 0.01, "xavier")

-- Train the network
for epoch = 1, 170000 do
    for i = 1, #xor_inputs do
        xor_net:backpropagate(xor_inputs[i], xor_outputs[i], activationFuncs)
    end

    -- Optional: Log outputs at certain epochs for debugging
    if epoch % 10000 == 0 then
        print("Epoch: " .. epoch)
        for i = 1, #xor_inputs do
            local outputs = xor_net:activate(xor_inputs[i], activationFuncs)
            print("Input: ", xor_inputs[i][1], xor_inputs[i][2])
            print("Output: ")
            printTable(outputs)
        end
    end
end

-- Test the network
print("XOR Network Outputs after training:")
for i = 1, #xor_inputs do
    local outputs = xor_net:activate(xor_inputs[i], activationFuncs)
    print("Input: ", xor_inputs[i][1], xor_inputs[i][2])
    print("Output: ")
    printTable(outputs)
end