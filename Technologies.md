# VLM - Vision Bridge 

## Technologies Used 
For this project, the technologies used for development are going to be:  
> TinyVLM 
>> A vision language model is a neural network architecture that bridges the gap between visual and textual information, enabling AI systems to understand, describe, and reason about images in natural language.  
>> More information about TinyVLM can be found at: https://github.com/anananan116/TinyVLM

> Python
>> Version 3.11 is used for developing the core logic and integration with TinyVLM  

> Docker
>> Docker will be used for the containerization of our AI services, aswell as for compatibility between different operating systems.  

## How these technologies will be used.

> TinyVLM
- The main part of this project, TinyVLM will be used to identify, and translate into auditory output the visual elements captured by the camera.
- It was chose instead of other VLM's due to the fact that TinyVLM is a compact model, specifically designed to interpret and describe the visual input using natural language.

> Python
- The primary development language for this project due to its extensive ecosystem of machine learning and AI libraries, readability, and rapid development capabilities. Its mature tooling around deep learning, computer vision, and data processing makes it ideal for integrating TinyVLM and handling real-time image analysis.

> Docker 
- Docker will be used to containerize the entire system, ensuring consistent performance and compatibility across different environments. Since our goal is to make the service portable and lightweight—capable of running on devices ranging from full desktop computers to low-power microprocessors—Docker provides a practical solution to handle cross-platform deployment.
By encapsulating dependencies and configurations in a single container, Docker eliminates the need for separate code versions for each operating system. This approach simplifies maintenance, improves reproducibility, and allows for faster testin





