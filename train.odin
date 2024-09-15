//+private file
package main

import "core:fmt"
import "core:time"
import rand "core:math/rand"
import "core:math"
import "core:os"
import "core:bufio"

// MARK: Load Mnist

@private
load_mnist_data :: proc(path: string, size: int) -> (ret: [dynamic]MnistRecord, ok: bool) {
    f, ferr := os.open(path)
    if ferr != 0 do return
    defer os.close(f)

    r: bufio.Reader 
    buffer: [1024]byte
    bufio.reader_init_with_buf(&r, os.stream_from_handle(f), buffer[:])
    defer bufio.reader_destroy(&r)

    // Ignore csv file header
    line, err := bufio.reader_read_string(&r, '\n', context.temp_allocator)

    i := 0
    ret = make([dynamic]MnistRecord, size)
    for { 
        defer i += 1 
        line, err := bufio.reader_read_string(&r, '\n', context.temp_allocator)
        if err != nil || i >= size - 1 { 
            break 
        } 

        // Process line
        values := split_u8_string(line)
        ret[i].label = values[0]
        for j in 1..=MNIST_IMG_DATA_LEN {
            ret[i].pixels[j-1] = f32 (values[j]) / 255.0
        }
    }

    return ret, true
}

// MARK: Train

train_step :: proc(net: ^Net, batch: []MnistRecord, learning_rate: f32) -> f32 {
    // Net struct temporarily stores the gradient values
    grad := Net{}
    net_init_mem(&grad, true)

    total_loss: f32
    batch_size := f32(len(batch))

    // Backpropagate, accumulate gradients for the entire batch
    for &img in batch {
        loss, _ := net_backward(net, &img, &grad, nil, true)
        total_loss += loss
    }
    
    // Adjust weights/bias based on averaged gradients
    for net_layer, i in &net.layers {
        grad_layer := &grad.layers[i]
        for j in 0..<len(net_layer.b) {
            net_layer.b[j] -= learning_rate * grad_layer.b[j] / batch_size
            for k in 0..<len(net_layer.w[j]) {
                net_layer.w[j][k] -= learning_rate * grad_layer.w[j][k] / batch_size
            }
        }
    }
    
    return total_loss / batch_size
}

@private
trian :: proc() {
    start_time := time.now()

    // Load mnist data
    fmt.println("Loading Training data ...")
    train_data, train_ok := load_mnist_data(MNIST_TRAIN_FILE_PATH, TRAIN_DATA_LEN)
    defer delete(train_data)
    if !train_ok {
        fmt.println("Failed to read mnist train data file")
        return
    }
    fmt.println("Loading Testing data ...")
    test_data, test_ok := load_mnist_data(MNIST_TEST_FILE_PATH, TEST_DATA_LEN)
    defer delete(test_data)
    if !test_ok {
        fmt.println("Failed to read mnist test data file")
        return
    }

    // Augment data
    fmt.println("Augmenting training data ...")
    for n in 0..=DATA_AUGMENTATION_COUNT {
        for i := TRAIN_DATA_LEN - 1; i >= 0; i -= 1 {
            record := train_data[i]
            augmented := augment_mnist_record(&record)
            append(&train_data, augmented)
        }
    }
    fmt.println("Done Augmenting, train data len:", len(train_data), ", test data len:", len(test_data))

    // Init Net
    net := Net{}
    net_init_mem(&net)
    net_init_values(&net)
    defer net_free(&net)

    batch_start := 0
    train_data_len := len(train_data)
    step := 0
    for step := 0; step < NUM_STEPS; step += 1 {
        defer batch_start = (batch_start + BATCH_SIZE) % train_data_len
        defer free_all(context.temp_allocator)

        batch := train_data[batch_start:batch_start+BATCH_SIZE]
        lr := f32(LEARNING_RATE)
        loss := train_step(&net, batch, lr)

        if step % 250 == 0 {
            accuracy := calc_net_accuracy(test_data[:], &net)
            fmt.println(
                "Step:", step, 
                "Accuracy:", accuracy, 
                "Learning Rate:", lr, 
                "Ts:", time.diff(start_time, time.now())
            )
        }
        if step % 2500 == 0 {
            if !net_save(&net) {
                fmt.println("Failed to save net")
            }
        }
    }
}

// MARK: Validate

calc_net_accuracy :: proc(test_dataset: []MnistRecord, net: ^Net) -> f32 {
    count := 0
    for &img in test_dataset {
        preds := net_forward(net, &img)
        prediction_idx := get_prediction_index(preds[len(preds) - 1])

        if prediction_idx == int (img.label) {
            count += 1
        }
    }

    return f32 (count) / f32 (len(test_dataset))
}

// MARK: Augmentation

augment_mnist_record :: proc(record: ^MnistRecord) -> (ret: MnistRecord) {
    ret.label = record.label
    augmentation := rand.int_max(2)

    switch augmentation {
    case 0:
        ret.pixels = rotate_image(record.pixels, rand.float32_range(-15, 15))
    // case 1:
    //     ret.pixels = add_noise(record.pixels, 0.1)
    case:
        sign1 := 1 if rand.int_max(2) == 0 else -1
        sign2 := 1 if rand.int_max(2) == 0 else -1
        ret.pixels = shift_image(record.pixels, sign1 * rand.int_max(3), sign2 * rand.int_max(3))
    }
    
    return ret
}

rotate_image :: proc(pixels: [MNIST_IMG_DATA_LEN]f32, angle: f32) -> [MNIST_IMG_DATA_LEN]f32 {
    result: [MNIST_IMG_DATA_LEN]f32
    center := f32(MNIST_IMG_SIZE / 2)
    angle_rad := angle * math.PI / 180
    cos_a := math.cos(angle_rad)
    sin_a := math.sin(angle_rad)
    
    for y in 0..<MNIST_IMG_SIZE {
        for x in 0..<MNIST_IMG_SIZE {
            new_x := int((f32(x) - center) * cos_a - (f32(y) - center) * sin_a + center)
            new_y := int((f32(x) - center) * sin_a + (f32(y) - center) * cos_a + center)
            if new_x >= 0 && new_x < MNIST_IMG_SIZE && new_y >= 0 && new_y < MNIST_IMG_SIZE {
                result[y * MNIST_IMG_SIZE + x] = pixels[new_y * MNIST_IMG_SIZE + new_x]
            }
        }
    }
    
    return result
}

shift_image :: proc(pixels: [MNIST_IMG_DATA_LEN]f32, dx: int, dy: int) -> [MNIST_IMG_DATA_LEN]f32 {
    result: [MNIST_IMG_DATA_LEN]f32
    
    for y in 0..<MNIST_IMG_SIZE {
        for x in 0..<MNIST_IMG_SIZE {
            new_x := x + dx
            new_y := y + dy
            
            if new_x >= 0 && new_x < MNIST_IMG_SIZE && new_y >= 0 && new_y < MNIST_IMG_SIZE {
                result[y * MNIST_IMG_SIZE + x] = pixels[new_y * MNIST_IMG_SIZE + new_x]
            }
        }
    }
    
    return result
}

add_noise :: proc(pixels: [MNIST_IMG_DATA_LEN]f32, noise_level: f32) -> [MNIST_IMG_DATA_LEN]f32 {
    result := pixels
    
    for i in 0..<MNIST_IMG_DATA_LEN {
        if pixels[i] > 0.0 {
            noise := rand.float32_range(-noise_level, noise_level)
            result[i] = clamp(result[i] + noise, 0, 1)
        }
    }
    
    return result
}

// MARK: Utils

@private
get_prediction_index :: proc(preds: []f32) -> int {
    prediction_idx := 0
    for i in 0..<MNIST_NUM_LABELS {
        if preds[i] > preds[prediction_idx] {
            prediction_idx = i
        }
    }
    return prediction_idx
}

pixel_to_f32 :: proc(value: u8) -> f32 {
    return f32 (value) / 255.0
}

string_to_u8 :: proc(s: string) -> u8 {
    result: u8 = 0;

    for ch in s {
        result = result * 10 + (u8(ch) - '0');
    }

    return result;
}

split_u8_string :: proc(s: string) -> (result: [MNIST_IMG_DATA_LEN + 1]u8) {
    item_index := 0
    start := 0
    str_len := len(s)

    i := 0
    for i < str_len {
        defer i += 1
        if s[i] == ',' || i == str_len - 1 {
            end := i

            // Include the last character in the line
            if i == str_len - 1 {
                end = i + 1
            }

            result[item_index] = string_to_u8(s[start:end])
            item_index += 1
            start = i + 1
        }
    }

    return result;
}

clamp :: proc(value, min, max: f32) -> f32 {
    if value < min do return min
    if value > max do return max
    return value
}
