//+private file
package main

import "core:fmt"
import "core:mem"
import rand "core:math/rand"
import "core:time"

check_memory_allocations :: proc(track: ^mem.Tracking_Allocator) {
    if len(track.allocation_map) > 0 {
        fmt.eprintf("=== %v allocations not freed: ===\n", len(track.allocation_map))
        for _, entry in track.allocation_map {
            fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
        }
    }
    if len(track.bad_free_array) > 0 {
        fmt.eprintf("=== %v incorrect frees: ===\n", len(track.bad_free_array))
        for entry in track.bad_free_array {
            fmt.eprintf("- %p @ %v\n", entry.memory, entry.location)
        }
    }
    mem.tracking_allocator_destroy(track)
}

run_viz :: proc() {
    // Viz Init
    err := viz_init()
    defer viz_deinit()
    if err { 
        return 
    }

    // Load test data
    test_data, test_ok := load_mnist_data(MNIST_TEST_FILE_PATH, TEST_DATA_LEN)
    defer delete(test_data)
    if !test_ok {
        fmt.println("Failed to read mnist test data file")
        return
    }

    // Sim loop
    frame_idx := 0
    img_idx := 0
    for {
        defer frame_idx = (frame_idx + 1) % 15
        if frame_idx == 0 {
            img_idx = (img_idx + 1) % TEST_DATA_LEN
        }
        if is_viz_terminate() {
            break
        }

        viz_update(&test_data[img_idx])
        free_all(context.temp_allocator)
    }

    free_all(context.temp_allocator)
}

@private
main :: proc() {
    // Debug mem track
    track: mem.Tracking_Allocator
    mem.tracking_allocator_init(&track, context.allocator)
    context.allocator = mem.tracking_allocator(&track)
    defer check_memory_allocations(&track)

    // seed rand
    rand.reset(u64(time.to_unix_nanoseconds(time.now())))

    // trian()
    run_viz()
}
