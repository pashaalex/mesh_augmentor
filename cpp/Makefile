# Makefile

TARGET_LIB = libmesh_render
SRC = mesh_render.cpp
OBJ = $(SRC:.cpp=.o)
HEADERS = mesh_render.h

# Windows
ifeq ($(OS),Windows_NT)
    TARGET = $(TARGET_LIB).dll
    CXXFLAGS += -DWIN32 -D_WINDOWS -W3 -m64 -DBUILD_DLL -fopenmp
    LDFLAGS += -shared -m64 -fopenmp
    CC = x86_64-w64-mingw32-g++
else
    # Linux
    TARGET = lib$(TARGET_LIB).so
    CXXFLAGS += -fPIC -Wall -m64 -fopenmp
    LDFLAGS += -shared -m64 -fopenmp
    CC = g++
endif

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)