set(TARGET llama-cli)
add_executable(${TARGET}
    cli.cpp
    ../server/server-task.cpp
    ../server/server-task.h
    ../server/server-queue.cpp
    ../server/server-queue.h
    ../server/server-common.cpp
    ../server/server-common.h
    ../server/server-context.cpp
    ../server/server-context.h
    )
target_link_libraries(${TARGET} PRIVATE common llama mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

include_directories(../server)
include_directories(../mtmd)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
