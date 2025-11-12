set(TARGET llama-cli)
add_executable(${TARGET}
    main.cpp
    ../server/server-http.h
    ../server/server-main.h
    ../server/server-main.cpp
    )
target_link_libraries(${TARGET} PRIVATE common llama mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ../server)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
