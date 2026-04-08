cmake_minimum_required(VERSION 3.14)
project("webgpu-bench" C CXX)

# llama.cpp lives as a git submodule in this repo
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" CACHE PATH "Path to llama.cpp source")

# JSPI vs Asyncify: JSPI for Chrome, Asyncify for Safari/Firefox
option(BENCH_USE_JSPI "Use JSPI instead of Asyncify for async WebGPU ops" ON)

# Disable parts of llama.cpp we don't need
set(LLAMA_BUILD_SERVER  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)

# Disable MEMORY64 for cross-browser compatibility (wllama does the same)
# Memory64 has issues with Asyncify on Firefox/Safari
set(LLAMA_WASM_MEM64    OFF CACHE BOOL "" FORCE)

# Include llama.cpp as subdirectory — builds ggml + llama as libraries
add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

# Our benchmark executable
add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE llama ggml)
target_include_directories(bench PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
)

if (EMSCRIPTEN)
    set_target_properties(bench PROPERTIES
        SUFFIX ".js"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Common link options for both variants
    target_link_options(bench PRIVATE
        # Exported C functions callable from JS
        "SHELL:-sEXPORTED_FUNCTIONS=_bench_init,_bench_load,_bench_run,_bench_exit,_malloc,_free"
        "SHELL:-sEXPORTED_RUNTIME_METHODS=ccall,cwrap,FS,MEMFS,HEAPU8,UTF8ToString"
        # Module pattern: createBenchModule() factory function
        "SHELL:-sMODULARIZE=1"
        "SHELL:-sEXPORT_NAME=createBenchModule"
        # Memory
        "SHELL:-sINITIAL_MEMORY=128MB"
        "SHELL:-sMAXIMUM_MEMORY=4096MB"
        "SHELL:-sALLOW_MEMORY_GROWTH=1"
        # Filesystem (for model files)
        "SHELL:-sFORCE_FILESYSTEM=1"
        # Keep runtime alive
        "SHELL:-sNO_EXIT_RUNTIME=1"
        # No main entry point
        "--no-entry"
    )

    # JSPI vs Asyncify specific flags
    if (BENCH_USE_JSPI)
        message(STATUS "webgpu-bench: using JSPI async mode")
        target_link_options(bench PRIVATE
            "SHELL:-sJSPI"
            "SHELL:-sJSPI_EXPORTS=['bench_init','bench_load','bench_run','bench_exit']"
            "-fwasm-exceptions"
        )
    else()
        message(STATUS "webgpu-bench: using Asyncify async mode")
        target_link_options(bench PRIVATE
            "SHELL:-sASYNCIFY=1"
            "-fexceptions"
        )
    endif()

    # Optimization flags
    target_compile_options(bench PRIVATE -O3 -msimd128 -flto=full -DNDEBUG)
endif()
