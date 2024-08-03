# Vulkan Project

## Overview

This project is a Vulkan application built following a comprehensive tutorial. The goal of the tutorial was to introduce key Vulkan concepts and provide practical experience with the API. 

## Concepts Covered

The tutorial explained the following Vulkan concepts (https://vulkan-tutorial.com/):

- **Dynamic Uniforms**: Allows updating uniform data dynamically at runtime.
- **Separate Images and Sampler Descriptors**: Separates image and sampler resources, improving flexibility and performance.
- **Pipeline Cache**: Caches compiled pipeline state objects to accelerate pipeline creation.
- **Multi-threaded Command Buffer Generation**: Enables command buffer generation across multiple threads for better performance.
- **Multiple Subpasses**: Utilizes multiple subpasses within a render pass to optimize rendering operations.

## Project Structure

- **VulkanSetup**: Handles the Vulkan initialization and setup.
- **QueueFamilyIndices**: Defines and uses structures related to Vulkan queue families.
- **Shaders**: Includes vertex and fragment shaders used in the project.
- **Model Loading**: Demonstrates loading models with multiple textures.

## Screenshots

A screenshot showcasing the project's output can be found below:

![Screenshot](screenshots/screenshot_1.png)

## Building and Running

To build and run the project, follow the instructions provided in the `Makefile`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
