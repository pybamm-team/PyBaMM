vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO LLNL/sundials
    REF v7.6.0
    SHA512 b6d15f68f25c5326bd42abb5e3652cc98e83d2eb31b213c9144b46c5b93fd123be5972e9d36217fdd09a0002dee3f78e530c21eda85f3b4d1d8d93b007546ea0
    HEAD_REF master
    PATCHES
        find-klu.patch
    GITHUB_HOST https://github.com
)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "static" SUN_BUILD_STATIC)
string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "dynamic" SUN_BUILD_SHARED)

if ("klu" IN_LIST FEATURES)
  set(ENABLE_KLU ON)
else()
  set(CMAKE_DISABLE_FIND_PACKAGE_SUITESPARSE ON)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DEXAMPLES_ENABLE_C=OFF
        -DEXAMPLES_ENABLE_CXX=OFF
        -DBUILD_STATIC_LIBS=${SUN_BUILD_STATIC}
        -DBUILD_SHARED_LIBS=${SUN_BUILD_SHARED}
        -DENABLE_KLU=${ENABLE_KLU}
        -DENABLE_OPENMP=ON
)

vcpkg_install_cmake(DISABLE_PARALLEL)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

file(GLOB REMOVE_DLLS
    "${CURRENT_PACKAGES_DIR}/debug/lib/*.dll"
    "${CURRENT_PACKAGES_DIR}/lib/*.dll"
)

file(GLOB DEBUG_DLLS
    "${CURRENT_PACKAGES_DIR}/debug/lib/*.dll"
)

file(GLOB DLLS
    "${CURRENT_PACKAGES_DIR}/lib/*.dll"
)

if(DLLS)
    file(INSTALL ${DLLS} DESTINATION ${CURRENT_PACKAGES_DIR}/bin)
endif()

if(DEBUG_DLLS)
    file(INSTALL ${DEBUG_DLLS} DESTINATION ${CURRENT_PACKAGES_DIR}/debug/bin)
endif()

file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
file(REMOVE "${CURRENT_PACKAGES_DIR}/LICENSE")
file(REMOVE "${CURRENT_PACKAGES_DIR}/debug/LICENSE")

if(REMOVE_DLLS)
    file(REMOVE ${REMOVE_DLLS})
endif()

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/${PORT}")
