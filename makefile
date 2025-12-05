FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile
GCC   = gcc
NVCC = nvcc

nbody: nbody.o compute.o
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(GCC) $(FLAGS) -c $< -o $@
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) -x cu -c $< -o $@
clean:
	rm -f *.o nbody 
