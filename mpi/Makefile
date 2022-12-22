TARGET = out
SOURCE = main.cpp my_mpi.cpp
PROCS  = 4
NUMS   = 1000


build:
	rm -f $(TARGET)
	mpic++ $(SOURCE) -o $(TARGET) -g3 -ggdb -std=c++11

run:
	mpirun -n $(PROCS) ./$(TARGET) $(NUMS)

