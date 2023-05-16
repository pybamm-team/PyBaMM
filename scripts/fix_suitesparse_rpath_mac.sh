#!/usr/bin/env bash

LIBDIR=${HOME}/.local/lib

otool -L ${LIBDIR}/libklu.2.dylib

install_name_tool -change @rpath/libsuitesparseconfig.6.dylib ${LIBDIR}/libsuitesparseconfig.6.dylib ${LIBDIR}/libklu.2.dylib

install_name_tool -change @rpath/libamd.3.dylib ${LIBDIR}/libamd.3.dylib ${LIBDIR}/libklu.2.dylib
install_name_tool -change @rpath/libcolamd.3.dylib ${LIBDIR}/libcolamd.3.dylib ${LIBDIR}/libklu.2.dylib
install_name_tool -change @rpath/libbtf.2.dylib ${LIBDIR}/libbtf.2.dylib ${LIBDIR}/libklu.2.dylib

install_name_tool -change @rpath/libsuitesparseconfig.6.dylib ${LIBDIR}/libsuitesparseconfig.6.dylib ${LIBDIR}/libcolamd.3.dylib
install_name_tool -change @rpath/libsuitesparseconfig.6.dylib ${LIBDIR}/libsuitesparseconfig.6.dylib ${LIBDIR}/libbtf.2.dylib
install_name_tool -change @rpath/libsuitesparseconfig.6.dylib ${LIBDIR}/libsuitesparseconfig.6.dylib ${LIBDIR}/libcolamd.3.dylib

otool -L ${LIBDIR}/libklu.2.dylib
