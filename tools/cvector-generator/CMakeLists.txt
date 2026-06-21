set(TARGET llama-cvector-generator)
add_executable(${TARGET} cvector-generator.cpp pca.hpp)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../server)
target_link_libraries(${TARGET} PUBLIC server-context llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
