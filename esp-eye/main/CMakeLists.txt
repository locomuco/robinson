idf_component_register(SRCS "main.c"
                       REQUIRES esp_driver_uart esp_psram
                       PRIV_REQUIRES esp_timer)
