/usr/bin/c++  -DSRM_EXPORTS  -std=c++1z -Wall -fopenmp -fPIC   -std=gnu++17 -o segment.cpp.o -c segment.cpp
/usr/bin/c++  -DSRM_EXPORTS  -std=c++1z -Wall -fopenmp -fPIC   -std=gnu++17 -o imageregion.cpp.o -c src/imageregion.cpp
/usr/bin/c++  -DSRM_EXPORTS  -std=c++1z -Wall -fopenmp -fPIC   -std=gnu++17 -o srm.cpp.o -c src/srm.cpp

/usr/bin/c++ -fPIC  -std=c++1z -Wall -fopenmp  -shared  -o libPIMZ.so  src/srm.cpp src/imageregion.cpp segment.cpp -lz

