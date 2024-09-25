//+private file
package main

import "core:fmt"
import rl "vendor:raylib"

// MARK: Consts

CHECKBOX_SIZE :: 20
BUTTON_PADDING :: 15
FONT_SIZE :: 20
WIDGET_HEIGHT :: 20

TAB_HEIGHT :: 30
TAB_LEFT_PADDING :: 30

// MARK: Structs

CheckBox :: struct {
    label: string,
    is_enabled: ^bool,
}

Button :: struct {
    label: string,
    on_click: proc()
}

Slider :: struct {
    label: string,
    min, max: f32,
    value: ^f32,
    is_updating: ^bool,
}

Widget :: union {
    CheckBox,
    Button,
    Slider,
}

Container :: struct {
    label: string,
    pos: rl.Vector2,
    widgets: []Widget,
    is_visible: bool
}

DropdownList :: struct {
    containers: []Container,
    active_menu_index: ^int,
    pos: rl.Vector2,
}

// MARK: Gui

@private
show_gui :: proc(flags: ^Flags, thresholds: ^Thresholds, reset_cam: proc()) {
    // TODO this is a bad way of doing this, and i can't think of anything better
    // The sliders need a boolean variable to track when its being updated
    // Else the slider updates are janky
    @static active_menu_index := -1
    @static slider_updates: struct {
        weight_cloud: bool,
        activation: bool,
        connection: bool,
    }

    // Construct menu structure
    controls := Container {
        label = "Controls",
        widgets = {
            CheckBox { "Rotate Cam", &flags.cam_rotate },
            CheckBox { "Show Cube Lines", &flags.draw_cube_lines },
            CheckBox { "Show Cubes", &flags.draw_cubes },
            CheckBox { "Show Weight Cloud", &flags.draw_weight_cloud },
            CheckBox { "Show Node Activations", &flags.draw_node_activations },
            CheckBox { "Show Connections", &flags.draw_connections },
            CheckBox { "Load Test Images", &flags.load_test_imgs },
            Button { "Reset Camera", reset_cam }
        },
    }
    thresholds := Container {
        label = "Thresholds",
        widgets = {
            Slider { "Weight Cloud", 0, 100, &thresholds.weight_cloud, &slider_updates.weight_cloud },
            Slider { "Activation", 0, 100, &thresholds.activations, &slider_updates.activation },
            Slider { "Connection", 0, 100, &thresholds.connections, &slider_updates.connection },
        },
    }
    menu := DropdownList {
        pos = {20, 20},
        containers = {
            controls, thresholds
        },
        active_menu_index = &active_menu_index
    }

    ui_dropdown_list(&menu)
}

// MARK: Menu

ui_dropdown_list :: proc(menu: ^DropdownList) {
    total_height: f32 = 0
    for &container, i in menu.containers {
        is_active := i == menu.active_menu_index^
        label_fmt := "< %s" if is_active else "> %s"
        label := cstring(raw_data(fmt.tprintf(label_fmt, container.label)))
        text_width := rl.MeasureText(label, FONT_SIZE)
        
        rl.DrawText(label, i32(menu.pos.x) + 5, i32(menu.pos.y + total_height) + 5, FONT_SIZE, rl.WHITE)
        
        is_mouse_on_area := rl.CheckCollisionPointRec(
            rl.GetMousePosition(), 
            {menu.pos.x, menu.pos.y + total_height, f32(text_width + 10), f32(TAB_HEIGHT)}
        )
        if is_mouse_on_area && rl.IsMouseButtonPressed(.LEFT) {
            menu.active_menu_index^ = i == menu.active_menu_index^ ? -1 : i
        }
        
        total_height += TAB_HEIGHT
        if is_active {
            content_pos_x := menu.pos.x + TAB_LEFT_PADDING
            content_pos_y := menu.pos.y + total_height + WIDGET_HEIGHT
            container.pos = {content_pos_x, content_pos_y}
            
            ui_container(&container)
            height := calculate_container_height(&container)
            total_height += f32(height) + WIDGET_HEIGHT * 2
        }
    }
}

// MARK: Container

ui_container :: proc(container: ^Container) {
    posx := i32(container.pos.x)
    posy := i32(container.pos.y)
    label := cstring(raw_data(container.label))
    for &w in container.widgets {
        switch &widget in w {
            case Button:
                posy += 5
                ui_button(&widget, posx, posy)
                posy += WIDGET_HEIGHT
            case CheckBox:
                ui_checkbox(&widget, posx, posy)
                posy += WIDGET_HEIGHT / 2
            case Slider:
                ui_slider(&widget, posx, posy)
                posy += WIDGET_HEIGHT * 2
        }
        posy += WIDGET_HEIGHT
    }
}

// MARK: Button

ui_button :: proc(button: ^Button, posx, posy: i32) {
    posx := posx + BUTTON_PADDING / 2
    label := cstring(raw_data(button.label))

    text_width := rl.MeasureText(label, FONT_SIZE)
    total_width := f32(text_width + BUTTON_PADDING)
    total_height := f32(FONT_SIZE + BUTTON_PADDING)
    is_mouse_on_area := rl.CheckCollisionPointRec(
        rl.GetMousePosition(), 
        {f32(posx), f32(posy), total_width, total_height}
    )

    if is_mouse_on_area {
        rl.DrawRectangle(
            posx - BUTTON_PADDING/2, posy - BUTTON_PADDING/2, 
            i32(total_width), i32(total_height), rl.ColorAlpha(rl.GRAY, 0.5)
        )
    }
    rl.DrawRectangleLines(
        posx - BUTTON_PADDING/2, posy - BUTTON_PADDING/2, 
        i32(total_width), i32(total_height), rl.WHITE
    )
    rl.DrawText(label, posx, posy, FONT_SIZE, rl.WHITE)

    if is_mouse_on_area && rl.IsMouseButtonPressed(.LEFT) {
        button.on_click()
    }
}

// MARK: Checkbox

ui_checkbox :: proc(checkbox: ^CheckBox, posx, posy: i32) {
    checkbox_enabled_size: i32 = CHECKBOX_SIZE - 6
    enabled_size_diff: i32 = (CHECKBOX_SIZE - checkbox_enabled_size) / 2

    label := cstring(raw_data(checkbox.label))
    text_width := rl.MeasureText(label, FONT_SIZE)
    
    // Draw checkbox
    rl.DrawRectangleLines(
        posx, posy, 
        CHECKBOX_SIZE, CHECKBOX_SIZE, 
        rl.WHITE
    )
    if checkbox.is_enabled^ {
        rl.DrawRectangle(
            i32(posx) + enabled_size_diff, i32(posy) + enabled_size_diff, 
            checkbox_enabled_size, checkbox_enabled_size, 
            rl.WHITE
        )
    }
    
    // Draw text
    text_pos_x := posx + CHECKBOX_SIZE + 10
    text_pos_y := posy + CHECKBOX_SIZE / 2 - 10
    rl.DrawText(label, text_pos_x, text_pos_y, FONT_SIZE, rl.WHITE)
    
    total_width := f32(CHECKBOX_SIZE + 10 + text_width)
    total_height := f32(max(CHECKBOX_SIZE, FONT_SIZE))
    is_mouse_on_area := rl.CheckCollisionPointRec(
        rl.GetMousePosition(), 
        {f32(posx), f32(posy), total_width, total_height}
    )
    if is_mouse_on_area && rl.IsMouseButtonPressed(.LEFT) {
        checkbox.is_enabled^ = !checkbox.is_enabled^
    }
}

// MARK: Slider

ui_slider :: proc(slider: ^Slider, posx, posy: i32) {
    SLIDER_W :: 200
    SLIDER_H :: 20
    LABEL_HEIGHT :: 20
    KNOB_SIZE :: SLIDER_H
    VALUE_WIDTH :: 50
    
    label := cstring(raw_data(fmt.tprintf("%s - %.1f", slider.label, slider.value^)))
    
    // Draw label
    rl.DrawText(label, posx, posy, FONT_SIZE, rl.WHITE)
    
    // Calculate positions
    slider_y := posy + LABEL_HEIGHT
    slider_end_x := posx + SLIDER_W
    
    // Draw slider bar
    rl.DrawRectangleLines(posx, slider_y, SLIDER_W, SLIDER_H, rl.WHITE)
    
    // Calculate and constrain knob position
    knob_pos_x := posx + i32((slider.value^ - slider.min) / (slider.max - slider.min) * f32(SLIDER_W - KNOB_SIZE))
    knob_pos_x = i32(clamp(f32(knob_pos_x), f32(posx), f32(slider_end_x - KNOB_SIZE)))
    knob_pos_y := slider_y + SLIDER_H/2 - KNOB_SIZE/2
    
    // Draw progress
    rl.DrawRectangle(posx, slider_y, knob_pos_x - posx + KNOB_SIZE/2, SLIDER_H, rl.WHITE)
    // Draw knob
    rl.DrawRectangle(knob_pos_x, knob_pos_y, KNOB_SIZE, KNOB_SIZE, rl.WHITE)
    
    // Handle mouse input
    mouse_pos := rl.GetMousePosition()
    slider_rect := rl.Rectangle{f32(posx), f32(slider_y), f32(SLIDER_W), f32(SLIDER_H)}
    
    // Check if the mouse is pressed on the slider
    if rl.CheckCollisionPointRec(mouse_pos, slider_rect) && rl.IsMouseButtonPressed(.LEFT) {
        slider.is_updating^ = true
    }
    
    // Update the value if the slider is active (being dragged)
    if slider.is_updating^ {
        if rl.IsMouseButtonDown(.LEFT) {
            new_value := slider.min + (slider.max - slider.min) * clamp(mouse_pos.x - f32(posx), 0, f32(SLIDER_W)) / f32(SLIDER_W)
            slider.value^ = clamp(new_value, slider.min, slider.max)
        } else {
            // Deactivate the slider when the mouse button is released
            slider.is_updating^ = false
        }
    }
}

// MARK: Utils

clamp :: proc(value, min, max: f32) -> f32 {
    if value < min do return min
    if value > max do return max
    return value
}

calculate_container_height :: proc(container: ^Container) -> i32 {
    height: i32 = 0
    for widget in container.widgets {
        switch w in widget {
            case Slider:
                height += WIDGET_HEIGHT * 3
            case CheckBox:
                height += WIDGET_HEIGHT + WIDGET_HEIGHT / 2
            case Button:
                height += WIDGET_HEIGHT
        }
    }
    return height
}
