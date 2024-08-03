# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Iinclude
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
SHADER_DIR = shaders

# Source files
SRC_FILES = $(SRC_DIR)/main.cpp $(SRC_DIR)/application.cpp $(SRC_DIR)/window.cpp $(SRC_DIR)/vulkanSetup.cpp

# Object files
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

# Shader files
SHADER_FILES = $(SHADER_DIR)/frag.spv $(SHADER_DIR)/vert.spv

# Release build flags
CFLAGS_RELEASE = -O2
# Debug build flags
CFLAGS_DEBUG = -g

# Application name
APP_NAME = Driewer_Vulkan

# Default to release build
all: $(BIN_DIR)/$(APP_NAME)

# Create object directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Create binary directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link object files into the release binary
$(BIN_DIR)/$(APP_NAME): $(OBJ_FILES) | $(BIN_DIR)
	$(CXX) $(CFLAGS_RELEASE) -o $@ $^ $(LDFLAGS)

# Link object files into the debug binary
$(BIN_DIR)/$(APP_NAME)_debug: $(OBJ_FILES)
	$(CXX) $(CFLAGS_DEBUG) -o $@ $^ $(LDFLAGS)

# Test the release build
test: $(BIN_DIR)/$(APP_NAME)
	./$(BIN_DIR)/$(APP_NAME)

# Test the debug build
test_debug: $(BIN_DIR)/$(APP_NAME)_debug
	./$(BIN_DIR)/$(APP_NAME)_debug

# Clean up build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Ensure shaders are up to date (not copied, just to ensure they are present)
.PHONY: shaders
shaders:
	@echo "Shaders are in $(SHADER_DIR)"

.PHONY: all test test_debug clean