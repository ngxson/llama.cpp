find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(IOSURFACE_LIBRARY IOSurface  REQUIRED)

ggml_add_backend_library(ggml-ane
                         ggml-ane.mm
                        )

target_compile_options(ggml-ane PRIVATE -fobjc-arc)
target_link_libraries(ggml-ane PRIVATE
                      ${FOUNDATION_LIBRARY}
                      ${IOSURFACE_LIBRARY}
                     )
