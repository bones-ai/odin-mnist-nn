package main

import "core:fmt"
import "core:os"
import "core:math"
import "core:encoding/json"
import rand "core:math/rand"

// MARK: Structs

// If a layer has n nodes and every node has m outputs,
// It has n bias and n*m weights
Layer :: struct {
    w: [][]f32,
    b: []f32,
    activation: Activation,
    dropout_rate: f32
}

Net :: struct {
    layers: []Layer
}

Activation :: enum {
    RELU,
    SOFTMAX
}

MnistRecord :: struct {
    pixels: [MNIST_IMG_DATA_LEN]f32,
    label: u8
}

// MARK: Utils

@(private="file")
get_rand_bias :: proc() -> f32 {
    // Bias is scaled down
    // Helps with stabalizing gradients at the start
    // Output: [0.0, 1.0) * 0.01
    return rand.float32() * 0.01
}

@(private="file")
get_rand_weight :: proc(num_inputs: f32, num_nodes: f32) -> f32 {
    // Xavier/Glorot initialization
    // Sets weights in the range [-limit, limit]
    // Maintains variance in activations and gradients
    // Which helps prevent vanishing and exploding gradients
    limit := math.sqrt(6.0 / (num_inputs + num_nodes))
    return rand.float32_range(-limit, limit)
}

// MARK: Activations

relu :: proc(values: []f32) {
    for &v in values {
        v = max(0, v)
    }
}

relu_derivative :: proc(values: []f32) {
    for &v in values {
        v = v > 0 ? 1 : 0
    }
}

softmax :: proc(values: []f32) {
    // Normalize layer ouptuts
    // Convert logits to probabilities
    assert(len(values) > 0)

    max_val := values[0]
    for i in values[1:] {
        max_val = max(max_val, i)
    }

    sum: f32
    for &i in values {
        i = math.exp(i - max_val)
        sum += i
    }

    for &i in values {
        i = i / sum
    }
}

// MARK: Layer

@(private="file")
layer_forward :: proc(layer: ^Layer, input: []f32, is_train: bool, contribs: ^Layer = nil) -> []f32 {
    output := make([]f32, len(layer.b), context.temp_allocator)
    for i in 0..<len(output) {
        output[i] = layer.b[i]
        for j in 0..<len(input) {
            output[i] += layer.w[i][j] * input[j]
            if contribs != nil {
                contribs.w[i][j] = math.abs(layer.w[i][j] * input[j])
            }
        }
    }
    
    switch layer.activation {
        case .RELU:
            relu(output)
        case .SOFTMAX:
            softmax(output)
    }

    // Dropout
    if is_train && layer.dropout_rate > 0 {
        for &v in output {
            if rand.float32() < layer.dropout_rate {
                v = 0
            } else {
                // Scale to maintain expected value
                v /= (1 - layer.dropout_rate) 
            }
        }
    }
    
    return output
}

// MARK: Net

net_init_mem :: proc(net: ^Net, use_temp_allocator: bool = false) {
    // The arch doesn't include the input and output layer
    // If each hidden layer has n nodes, and m inputs
    // m == number of nodes in the prev layer
    // n == number of outputs from the current layer
    // Weights size - [n][m], each node n has m connections
    // Bias size - n

    // Add the output layer
    arch := make([]u32, len(NET_ARCH) + 1, context.temp_allocator)
    copy(arch, NET_ARCH)
    arch[len(arch) - 1] = MNIST_NUM_LABELS

    // Build layers
    allocator := context.temp_allocator if use_temp_allocator else context.allocator
    net.layers = make([]Layer, len(arch), allocator)
    num_inputs := u32(MNIST_IMG_DATA_LEN)
    for num_nodes, i in arch {
        net.layers[i].b = make([]f32, num_nodes, allocator)
        net.layers[i].w = make([][]f32, num_nodes, allocator)
        for j in 0..<num_nodes {
            net.layers[i].w[j] = make([]f32, num_inputs, allocator)
        }

        net.layers[i].activation = .RELU
        if i == len(arch) - 1 {
            net.layers[i].activation = .SOFTMAX
        }
        num_inputs = num_nodes
    }
}

// Randomize the network weights/bias
net_init_values :: proc(net: ^Net) {
    num_inputs := u32(MNIST_IMG_DATA_LEN)
    for num_nodes, i in NET_ARCH {
        for j in 0..<num_nodes {
            net.layers[i].b[j] = get_rand_bias()
            for k in 0..<num_inputs {
                net.layers[i].w[j][k] = get_rand_weight(f32(num_inputs), f32(num_nodes))
            }
        }
        net.layers[i].activation = .RELU
        net.layers[i].dropout_rate = DROPOUT_RATE
        num_inputs = num_nodes
    }
    
    // Output layer
    last_layer := &net.layers[len(net.layers)-1]
    for j in 0..<MNIST_NUM_LABELS {
        last_layer.b[j] = get_rand_bias()
        for k in 0..<num_inputs {
            last_layer.w[j][k] = get_rand_weight(f32(num_inputs), f32(MNIST_NUM_LABELS))
        }
    }
    last_layer.activation = .SOFTMAX
}

net_forward :: proc(net: ^Net, img: ^MnistRecord, contribs: ^Net = nil, is_train: bool = false) -> [][]f32 {
    activations := make([][]f32, len(net.layers) + 1, context.temp_allocator)
    
    activations[0] = img.pixels[:]
    for i in 0..<len(net.layers) {
        layer := &net.layers[i]
        contrib := &contribs.layers[i]
        act := layer_forward(layer, activations[i], is_train, contrib)
        activations[i + 1] = act
    }

    return activations
}

net_backward :: proc(net: ^Net, img: ^MnistRecord, grad: ^Net, contribs: ^Net = nil, is_train: bool = false) -> (f32, [][]f32) {
    // Forward pass
    activations := net_forward(net, img, contribs, is_train)
    
    // Compute the output error based on predictions
    output_error := make([]f32, len(activations[len(activations)-1]), context.temp_allocator)
    for i in 0..<len(output_error) {
        output_error[i] = activations[len(activations)-1][i]
        if i == int(img.label) {
            output_error[i] -= 1
        }
    }
    loss := -math.ln(activations[len(activations)-1][img.label])
    
    // Backpropagation: 
    // Compute gradients for each layer
    for i := len(net.layers) - 1; i >= 0; i -= 1 {
        layer := &net.layers[i]
        grad_layer := &grad.layers[i]
        prev_act := activations[i]
        
        for j in 0..<len(layer.b) {
            grad_layer.b[j] += output_error[j]
            for k in 0..<len(prev_act) {
                grad_layer.w[j][k] += output_error[j] * prev_act[k]
            }
        }

        if i == 0 do continue
        
        // Compute the error for the previous layer
        prev_error := make([]f32, len(prev_act), context.temp_allocator)
        for j in 0..<len(prev_act) {
            for k in 0..<len(output_error) {
                prev_error[j] += output_error[k] * layer.w[k][j]
            }
        }
        if net.layers[i-1].activation == .RELU {
            relu_derivative(prev_act)
            for j in 0..<len(prev_error) {
                prev_error[j] *= prev_act[j]
            }
        }
        output_error = prev_error
    }
    
    return loss, activations
}

net_free :: proc(net: ^Net) {
    if net == nil do return

    for layer in net.layers {
        for w in layer.w {
            delete(w)
        }
        delete(layer.w)
        delete(layer.b)
    }

    delete(net.layers)
    net.layers = nil
}

// MARK: Save/Load

net_save :: proc(net: ^Net) -> bool {
    if data, err := json.marshal(net^, allocator = context.temp_allocator); err == nil {
        // Create the directory if it doesn't exist
        err := os.make_directory(NETWORK_SAVE_DIRECTORY)
        if os.write_entire_file(NETWORK_SAVE_FILE_PATH, data) {
            return true
        }
    }

    return false
}

// Do not init mem when loading a network
// It leads to memory leaks
net_load :: proc(net: ^Net) -> (err: bool) {
    if json_data, ok := os.read_entire_file(NETWORK_LOAD_FILE_PATH, context.temp_allocator); ok {
        if json.unmarshal(json_data, net) == nil {
            return false
        } else {
            fmt.println("Failed to unmarshal JSON")
        }
    } else {
        fmt.println("Failed to load saved net")
    }

    return true
}
