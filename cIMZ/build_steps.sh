/usr/bin/clang++  -DSRM_EXPORTS  -std=c++1z -Wall -fopenmp -fPIC   -std=gnu++17 -o segment.cpp.o -c segment.cpp
/usr/bin/clang++  -DSRM_EXPORTS  -std=c++1z -Wall -fopenmp -fPIC   -std=gnu++17 -o imageregion.cpp.o -c src/imageregion.cpp
/usr/bin/clang++  -DSRM_EXPORTS  -std=c++1z -Wall -fopenmp -fPIC   -std=gnu++17 -o srm.cpp.o -c src/srm.cpp

/usr/bin/clang++ -fPIC  -std=gnu++17 -Wall -fopenmp  -shared  -o libPIMZ.so  src/srm.cpp src/imageregion.cpp segment.cpp -lz

