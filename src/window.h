#ifndef WINDOW_H
#define WINDOW_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

class Window {
public:
    void initWindow();
    bool shouldClose();
    void pollEvents();
    void cleanup();
    GLFWwindow* getWindow() const;

private:
    GLFWwindow* window;
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
};

#endif // WINDOW_H