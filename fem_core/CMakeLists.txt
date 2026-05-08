# =============================================================================
# CMakeLists.txt — сборка вычислительного ядра fem_core.
# -----------------------------------------------------------------------------
# Требования (раздел 3.1.6 пояснительной записки):
#   - C++17;
#   - OpenMP (обязательная зависимость);
#   - флаги оптимизации: -O3 (GCC/Clang) или /O2 (MSVC);
#   - выходной артефакт — разделяемая библиотека fem_core.so / fem_core.dll
#     без декорирования имён, чтобы Python-сторона могла загрузить её через
#     ctypes.CDLL и вызывать функции по их C-именам.
#
# Использование:
#   cd fem_core
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . -j
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(fem_core LANGUAGES CXX VERSION 1.0.0)

# -----------------------------------------------------------------------------
# Стандарт C++17 и видимость символов.
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# По умолчанию символы скрыты — экспортируем только помеченные FEM_API.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# -----------------------------------------------------------------------------
# Тип сборки по умолчанию — Release.
# Генераторы Visual Studio мультиконфигурационные: CMAKE_BUILD_TYPE там
# не работает; конфигурация задаётся при сборке (--config Release).
# -----------------------------------------------------------------------------
get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _isMultiConfig AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# MSVC: убираем /RTC1 из флагов Debug, чтобы избежать конфликта с /O2.
# /RTC1 несовместим с любой оптимизацией; CMake добавляет его автоматически
# через CMAKE_CXX_FLAGS_DEBUG.
# -----------------------------------------------------------------------------
if(MSVC)
    foreach(_flag_var
            CMAKE_CXX_FLAGS_DEBUG
            CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REPLACE "/RTC1" "" ${_flag_var} "${${_flag_var}}")
        string(REPLACE "/RTC2" "" ${_flag_var} "${${_flag_var}}")
        set(${_flag_var} "${${_flag_var}}" CACHE STRING "" FORCE)
    endforeach()
endif()

# -----------------------------------------------------------------------------
# OpenMP.
# На MSVC поддерживается через /openmp; если не найден — параллелизм
# отключится без ошибки сборки (производительность упадёт, корректность
# сохранится).
# -----------------------------------------------------------------------------
find_package(OpenMP)

# -----------------------------------------------------------------------------
# Исходники.
# -----------------------------------------------------------------------------
set(FEM_CORE_SOURCES
    mesh.cpp
    assembly.cpp
    solver.cpp
    flux.cpp
    api.cpp
)

set(FEM_CORE_HEADERS
    mesh.hpp
    assembly.hpp
    solver.hpp
    flux.hpp
    api.hpp
    sparse.hpp
)

# -----------------------------------------------------------------------------
# Цель — разделяемая библиотека.
# -----------------------------------------------------------------------------
add_library(fem_core SHARED ${FEM_CORE_SOURCES} ${FEM_CORE_HEADERS})

target_include_directories(fem_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(fem_core PUBLIC
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# Имя выходного файла — fem_core.so / fem_core.dll (без префикса lib на Linux).
set_target_properties(fem_core PROPERTIES
    PREFIX ""
    OUTPUT_NAME "fem_core"
    POSITION_INDEPENDENT_CODE ON
)

# На MSVC: копируем .dll рядом с .lib, чтобы Python мог найти через ctypes.
if(MSVC)
    set_target_properties(fem_core PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_CURRENT_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

# -----------------------------------------------------------------------------
# Флаги компиляции — раздельно для GCC/Clang и MSVC.
# Ключевой принцип: оптимизация (/O2, -O3) только через конфигурационные
# генераторные выражения, чтобы не конфликтовать с Debug-флагами (/RTC1).
# -----------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(fem_core PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
        $<$<CONFIG:Debug>:-O0 -g>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
    )
elseif(MSVC)
    target_compile_options(fem_core PRIVATE
        /W4             # предупреждения
        /utf-8          # исходники в UTF-8
        /wd4819         # подавить предупреждение о кодировке файла
        /wd4251         # подавить dll-интерфейс для STL-членов
        /EHsc           # стандартная обработка исключений C++
        # Оптимизация — только для Release и RelWithDebInfo.
        # ВАЖНО: /O2 несовместим с /RTC1 (Debug runtime checks),
        # поэтому задаём его только через генераторные выражения.
        $<$<CONFIG:Release>:/O2 /Ob2 /DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Ob1 /DNDEBUG>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
endif()

# -----------------------------------------------------------------------------
# Установка библиотеки рядом с Python-пакетом, чтобы core_bridge.py
# мог найти её автоматически.
# -----------------------------------------------------------------------------
install(TARGETS fem_core
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# -----------------------------------------------------------------------------
# Информация при настройке.
# -----------------------------------------------------------------------------
if(_isMultiConfig)
    message(STATUS "fem_core: generator (multi-config) — build with --config Release")
else()
    message(STATUS "fem_core: build type           = ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "fem_core: C++ compiler         = ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "fem_core: OpenMP found         = ${OpenMP_CXX_FOUND}")
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "fem_core: OpenMP не найден — параллелизм отключён, сборка продолжается")
endif()
message(STATUS "fem_core: output library       = fem_core${CMAKE_SHARED_LIBRARY_SUFFIX}")
