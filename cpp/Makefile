# Makefile

TARGET_LIB = mesh_render
SRC = mesh_render.cpp
OBJ = $(SRC:.cpp=.o)
HEADERS = mesh_render.h

# Windows
ifeq ($(OS),Windows_NT)
    TARGET = $(TARGET_LIB).dll
    CXXFLAGS += -DWIN32 -D_WINDOWS -m64 -DBUILD_DLL -fopenmp -Wall -O3
    LDFLAGS += -shared -m64 -fopenmp
    CC = x86_64-w64-mingw32-g++
else
    # Linux
    TARGET = $(TARGET_LIB).so
    CXXFLAGS += -fPIC -Wall -m64 -fopenmp -O3
    LDFLAGS += -shared -m64 -fopenmp
    CC = g++
endif

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.cpp $(HEADERS)
	$(CC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
