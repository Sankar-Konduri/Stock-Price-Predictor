cmake_minimum_required(VERSION 3.16)

project(Stock-Price-Predictor VERSION 0.1.0 LANGUAGES CXX)

# ---- Language standard -------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)          # -std=c++17 rather than -std=gnu++17

# Write build/compile_commands.json (lets VS Code IntelliSense find include paths).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MinGW Makefiles is a single-configuration generator: default to Debug.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# ---- Warnings (shared by every target) --------------------------------------
add_library(spp_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(spp_warnings INTERFACE -Wall -Wextra)
elseif(MSVC)
    target_compile_options(spp_warnings INTERFACE /W4)
endif()

# ---- Core library: everything except tests and the executable's main() ------
add_library(spp_core STATIC
    src/models/PredictionModel.cpp
    src/SMA/SMA.cpp
    src/WMA/WMA.cpp
    src/LinearRegression/LinearRegression.cpp
    src/ensemble/Ensemble.cpp
    src/ensemble/Weighting.cpp
    src/data/DataLoader.cpp
    src/evaluation/ModelEvaluator.cpp
    src/evaluation/AdaptiveEvaluator.cpp
    src/app/Application.cpp
)
target_include_directories(spp_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(spp_core PRIVATE spp_warnings)

# ---- The application ---------------------------------------------------------
add_executable(stock_predictor src/main.cpp)
target_link_libraries(stock_predictor PRIVATE spp_core spp_warnings)

# ---- Tests -------------------------------------------------------------------
option(SPP_BUILD_TESTS "Build the unit tests" ON)

if(SPP_BUILD_TESTS)
    enable_testing()

    # Each tests/<name>.cpp becomes an executable linked against spp_core.
    # Tests run from the project root so relative paths like
    # "data/historical_prices.csv" resolve.
    function(spp_add_test test_name)
        add_executable(${test_name} tests/${test_name}.cpp)
        target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
        target_link_libraries(${test_name} PRIVATE spp_core spp_warnings)
        add_test(NAME ${test_name}
                 COMMAND ${test_name}
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endfunction()

    foreach(t
            test_SMA
            test_WMA
            test_LinearRegression
            test_Ensemble
            test_DataLoader
            test_ModelEvaluator
            test_Weighting
            test_AdaptiveEvaluator
            test_Application)
        spp_add_test(${t})
    endforeach()
endif()