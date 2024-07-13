# Makefile

TARGET_LIB = libmesh_render
SRC = mesh_render.cpp
OBJ = $(SRC:.cpp=.o)
HEADERS = mesh_render.h

# Windows
ifeq ($(OS),Windows_NT)
    TARGET = $(TARGET_LIB).dll
    CXXFLAGS += -DWIN32 -D_WINDOWS -W3 -m64
    LDFLAGS += -shared -m64
else
    # Linux
    TARGET = $(TARGET_LIB).so
    CXXFLAGS += -fPIC -Wall -m64
    LDFLAGS += -shared -m64
endif

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)