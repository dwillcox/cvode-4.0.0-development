rm -rf builddir instdir
mkdir builddir
mkdir instdir
cd builddir
make clean
cmake -DCMAKE_INSTALL_PREFIX=/home/dwillcox/actual_home/CVODE/instdir -DEXAMPLES_INSTALL_PATH=/home/dwillcox/actual_home/CVODE/instdir/examples -DCUDA_ENABLE=ON -DEXAMPLES_ENABLE_CUDA=ON ../../cvode-4.0.0-development
make install
cd ..
