
#
# Main component of TF Micro project 'person_detection'.
#

idf_component_register(
    SRCS
        "main.cpp" "test_image.h"

    PRIV_REQUIRES 
        console
        static_images
        spi_flash
        esp_psram
    INCLUDE_DIRS ".")
