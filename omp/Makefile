CC      = g++
EXEC    = lab
FLAGS   = -std=c++11
LIBS    = -fopenmp
RELEASE = -O3
DEBUG   = -O0 -g3 -ggdb


release:
	$(CC) main.cpp $(FLAGS) $(RELEASE) -o $(EXEC) $(LIBS)

debug:
	$(CC) main.cpp $(FLAGS) $(DEBUG) -o $(EXEC) $(LIBS)

