#ifndef APPLICATION_H
#define APPLICATION_H

#include "window.h"
#include "vulkanSetup.h"

class Application {
public:
    void run();

private:
    Window window;
    VulkanSetup vulkanSetup;

    void init();
    void mainLoop();
    void cleanup();
};

#endif // APPLICATION_H