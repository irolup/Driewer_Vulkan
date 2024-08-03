#include "window.h"

void Window::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Don't create OpenGL context
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan - Charlie", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

bool Window::shouldClose() {
    return glfwWindowShouldClose(window);
}

void Window::pollEvents() {
    glfwPollEvents();
}

void Window::cleanup() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

GLFWwindow* Window::getWindow() const {
    return window;
}

void Window::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    // Handle framebuffer resize if needed
}