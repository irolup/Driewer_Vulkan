#include "application.h"

void Application::run() {
    init();
    mainLoop();
    cleanup();
}

void Application::init() {
    window.initWindow();
    vulkanSetup.initVulkan(window);
}

void Application::mainLoop() {
    while (!window.shouldClose()) {
        window.pollEvents();
        vulkanSetup.drawFrame(window);
    }
    vulkanSetup.waitIdle();
}

void Application::cleanup() {
    vulkanSetup.cleanup(window);
    window.cleanup();
}