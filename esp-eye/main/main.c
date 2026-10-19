/*
 * Robinson's eye: the ESP-EYE camera as a USB camera for the Raspberry Pi.
 *
 * After boot the console UART (USB, CP2102) switches to 921600 baud and
 * carries JPEG frames instead of log output. The Pi sends one-byte commands:
 *   'q'  stream small frames (QVGA) continuously - face tracking
 *   'h'  send one large frame (SVGA), then continue as before - "what do you see?"
 *   'p'  pause the stream
 * Each frame: "RBNF" | u32 length | u16 width | u16 height | JPEG (little endian).
 */

#include <string.h>

#include "driver/uart.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define UART_PORT   UART_NUM_0
#define UART_BAUD   921600

static const char *TAG = "eye";

static const camera_config_t cam_cfg = {    /* ESP-EYE pinout */
    .pin_pwdn = -1, .pin_reset = -1, .pin_xclk = 4,
    .pin_sccb_sda = 18, .pin_sccb_scl = 23,
    .pin_d7 = 36, .pin_d6 = 37, .pin_d5 = 38, .pin_d4 = 39,
    .pin_d3 = 35, .pin_d2 = 14, .pin_d1 = 13, .pin_d0 = 34,
    .pin_vsync = 5, .pin_href = 27, .pin_pclk = 25,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0, .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_SVGA,         /* allocate for the largest size used */
    .jpeg_quality = 12,
    .fb_count = 2,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_LATEST,
};

static void send_frame(const camera_fb_t *fb)
{
    uint8_t hdr[12] = {'R', 'B', 'N', 'F'};
    uint32_t len = fb->len;
    uint16_t w = fb->width, h = fb->height;
    memcpy(hdr + 4, &len, 4);
    memcpy(hdr + 8, &w, 2);
    memcpy(hdr + 10, &h, 2);
    uart_write_bytes(UART_PORT, hdr, sizeof(hdr));
    uart_write_bytes(UART_PORT, fb->buf, fb->len);
}

/* grab a frame at `size`; after a size switch drop the stale frames and give
 * auto exposure time to settle (it restarts dark at the new size) */
static camera_fb_t *grab(sensor_t *s, framesize_t size, framesize_t *current)
{
    if (size != *current) {
        s->set_framesize(s, size);
        *current = size;
        for (int i = 0; i < (size == FRAMESIZE_QVGA ? 2 : 6); i++) {
            camera_fb_t *old = esp_camera_fb_get();
            if (old) esp_camera_fb_return(old);
        }
    }
    return esp_camera_fb_get();
}

void app_main(void)
{
    esp_err_t err = esp_camera_init(&cam_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed: %s", esp_err_to_name(err));
        vTaskDelay(pdMS_TO_TICKS(2000));
        esp_restart();
    }
    sensor_t *s = esp_camera_sensor_get();
    s->set_vflip(s, 1);                   /* ESP-EYE mounts the sensor upside down */
    s->set_hmirror(s, 1);
    framesize_t current = FRAMESIZE_SVGA;

    ESP_LOGW(TAG, "camera ok, switching UART to %d baud for frames", UART_BAUD);
    vTaskDelay(pdMS_TO_TICKS(50));
    esp_log_level_set("*", ESP_LOG_NONE); /* the UART belongs to the frames now */
    uart_driver_install(UART_PORT, 1024, 64 * 1024, 0, NULL, 0);
    uart_set_baudrate(UART_PORT, UART_BAUD);

    bool streaming = false, want_high = false;
    for (;;) {
        uint8_t cmd;
        while (uart_read_bytes(UART_PORT, &cmd, 1, 0) == 1) {
            if (cmd == 'q') streaming = true;
            else if (cmd == 'p') streaming = false;
            else if (cmd == 'h') want_high = true;
        }
        if (!streaming && !want_high) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        camera_fb_t *fb = grab(s, want_high ? FRAMESIZE_SVGA : FRAMESIZE_QVGA, &current);
        want_high = false;
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        send_frame(fb);
        esp_camera_fb_return(fb);
        uart_wait_tx_done(UART_PORT, pdMS_TO_TICKS(2000));   /* never queue stale frames */
    }
}
