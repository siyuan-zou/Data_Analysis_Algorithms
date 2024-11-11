CXX = g++
CXXFLAGS = ${INCLUDES} -std=c++11 -O3
INCLUDES = -Igradinglib -Igrading -IDataset -IConfusionMatrix -IKernel -ISvm

.PHONY: all clean

all: grader

# TD-specific part

OBJECTS_SPECIFIC = Dataset.o ConfusionMatrix.o Kernel.o Svm.o
OBJECTS_SPECIFIC = Dataset/Dataset.cpp ConfusionMatrix/ConfusionMatrix.cpp Kernel/Kernel.cpp Svm/Svm.cpp

Dataset.o: Dataset/Dataset.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) Dataset/Dataset.cpp

ConfusionMatrix.o: ConfusionMatrix/ConfusionMatrix.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) ConfusionMatrix/ConfusionMatrix.cpp

Kernel.o: Kernel/Kernel.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) Kernel/Kernel.cpp

Svm.o: Svm/Svm.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) Svm/Svm.cpp

# Common part
SOURCES_COMMON = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp
OBJECTS_COMMON = gradinglib.o grading.o main.o

grader: $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC)
	$(CXX) $(CXXFLAGS) -o grader $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC)

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CXXFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CXXFLAGS) -o grading.o grading/grading.cpp 

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CXXFLAGS) -o main.o main.cpp

clean:
	rm -f grader *.o
