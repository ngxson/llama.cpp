set(TARGET llama-tts)
add_executable(${TARGET} tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# throwaway test driver, to be removed once the real helper lands
set(TARGET qwen3-tts-test)
add_executable(${TARGET} qwen3-tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
