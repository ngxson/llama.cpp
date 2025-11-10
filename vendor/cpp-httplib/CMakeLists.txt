set(TARGET cpp-httplib)

find_package(Threads REQUIRED)

add_library(${TARGET} STATIC httplib.cpp httplib.h)
if (NOT MSVC)
    # disable warnings in 3rd party code
    target_compile_options(${TARGET} PRIVATE -w)
endif()

target_link_libraries  (${TARGET} PRIVATE Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
