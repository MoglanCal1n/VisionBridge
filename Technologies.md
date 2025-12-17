
## Table of contents for the first lab
- [Base functionalities](#base-functionalities)
- [Problem description](#problem-description)
- [Related work on the problem](#related-work-on-the-problem)
- [Technologies used](#technologies-used)
- [Training data](#training-data)
- [Algorithm and Model used](#algorithm-and-model-used)

---

## Base functionalities

### **Real-time analysis of the surrounding environment:**
- #### Continuously processes visual input to understand the user’s immediate surroundings.
### **Generation of detailed, context-aware natural language descriptions:**
- #### Produces clear and relevant explanations of what’s happening in the scene.
### **Interpretation of complex environments and dynamic elements:**
- #### Recognizes and tracks multiple objects, movements, and interactions in real-world contexts.
### **Responsive visual question answering for situational awareness:**
- #### Allows users to ask questions about their surroundings and receive informative, voice-based responses.
### **Facial recognition:**
- #### Identifies and announces names of recognized people nearby.

[🔝 Back to Top](#table-of-contents-for-the-first-lab)

---

## Problem description

### What's our problem?
- #### Blind or visually impaired people face major difficulties in perceiving and understanding the visual environment around them. Daily activities such as walking in an unfamiliar space, identifying obstacles, recognizing objects or interpreting the visual context become processes dependent on external help. The lack of real-time visual feedback limits the autonomy, safety with our demographic.
### Why is the subject important?
### This solution has a significant social impact, contributing to: 
- #### Increasing the autonomy and safety of visually impaired people; 
- #### Social and digital integration of them; 
- #### Efficient use of modern AI models (TinyVLM) on resource-constrained devices, enabling implementation on mobile phones, wearables or smart glasses.
### What do we solve?
- #### The proposed application provides real-time intelligent visual assistance, generating detailed and context-sensitive descriptions based on video stream captured by a camera.
### Who are our users?
- #### People who are blind or partially sighted;
- #### Organizations and institutions offering technological support for social inclusion; 
- #### Researchers and developers interested in AI-based visual assistance.
### What's our input and output data?
- #### Input: Video feed from a camera (e.g., camera mounted on glasses or phone) and voice questions from the user.
- #### Output: Natural language generated descriptions (text audio playback), answers to visual questions, alerts about obstacles or dynamic elements in the environment.
### How do we measure our performance?
- #### Accuracy of object and action recognition (metrics such as mAP, precision, recall);
- #### Quality of natural language description (BLEU, METEOR, CIDEr, SPICE);
- #### Response time (latency between image capture and feedback generation)
- #### Subjective human evalutation - degree of perceived usefulness and user satisfaction in real tests;
- #### Efficiency on low-power devices (memory consumption, inference speed)

[🔝 Back to Top](#table-of-contents-for-the-first-lab)

---

## Related work on the problem

### 1. Vision-Language Models for Edge Networks: A Comprehensive Survey
- #### Surveys Vision-Language Models (VLMs) optimized for edge devices with limited resources.
- #### Discusses key optimizations: model compression (pruning, quantization, distillation), hardware accelerators (Edge TPU), and efficient architectures (MobileNetV3, lightweight Transformers).
- #### Emphasizes real-time inference, data privacy, and IoT integration for healthcare, surveillance, and autonomous systems.

### 2. Navigation with VLM Framework: Go to Any Language (NavVLM)
- #### Introduces NavVLM, a plug-and-play framework enabling robots to navigate to goals defined in natural language.
- #### Uses a lightweight VLM (e.g., minicpm-llama3-v2.5) for zero-shot visual reasoning.
- #### Integrates VLM guidance, SLAM, and path planning (FMM) for robust navigation.
- #### Achieves state-of-the-art success rates and supports open-ended, language-driven goals.

### 3. MoViNets: Mobile Video Networks for Efficient Video Recognition
- #### Proposes MoViNets, efficient 3D CNNs for online video recognition on mobile and IoT devices.
- #### Boosts efficiency through:
    - #### Neural Architecture Search (NAS) for balanced spatiotemporal design.
    - #### Stream Buffers for constant memory and causal inference.
    - #### Temporal Ensembles to recover lost precision.
- #### Example: MoViNet-A5-Stream matches X3D-XL accuracy using 80% fewer FLOPs and 65% less memory.

### 4. TinyVLM: Scaling Down Vision-Language Models for the Edge
- #### Presents TinyVLM, a 0.6B-parameter compact VLM designed for real-time edge deployment.
- #### Key features:
    - #### CNN-Guided Token Pooling (ConvPooler) reduces token load.
    - #### Visual Feature Enrichment fuses CNN and ViT embeddings.
    - #### Patch Zooming captures multiscale detail.
- #### Trained in ~106 GPU hours and achieves 18 tokens/sec on an 8-core CPU, showing feasibility for low-power devices.

### 5. An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale (ViT)
- #### Introduces the Vision Transformer (ViT) — a Transformer-based image classifier using patch embeddings.
- #### Shows that large-scale pretraining enables ViTs to match or surpass CNNs.
- #### Key insights:
    - #### Needs large datasets (ImageNet-21k, JFT-300M).
    - #### Offers scalability and lower pretraining cost.
    - #### Tends to overfit small datasets, but generalizes well when scaled.

[🔝 Back to Top](#table-of-contents-for-the-first-lab)  

## Technologies used 
For this project, the technologies used for development are going to be:  
> moondream 
>> A vision language model is a neural network architecture that bridges the gap between visual and textual information, enabling AI systems to understand, describe, and reason about images in natural language.  
>> More information about moondream can be found at: <https://github.com/vikhyat/moondream>  

> Python
>> Version 3.11 is used for developing the core logic and integration with TinyVLM  

> Docker
>> Docker will be used for the containerization of our AI services, aswell as for compatibility between different operating systems.  

[🔝 Back to Top](#table-of-contents-for-the-first-lab)

##  Training Data
The training data set used for fine-tuning the model is available at: <https://universe.roboflow.com/ece479/crosswalk-qj5uo/>  
It is composed of approx. 1200 images divided as: 
> 70 images reserved for testing  
> 770 images reserved for training  
> 230 images reserved for validation  

The dataset has been downloaded from <https://universe.roboflow.com/ece479/crosswalk-qj5uo/> and consists of images depicting crosswalks captured under diverse conditions: 
> Varying resolutions  
> Lighting difference  
> Difference shutter speed, depth, WB and depth  
All the identifiable information, such as faces and car license plates have beeen blurred to preserve privacy.

[🔝 Back to Top](#table-of-contents-for-the-first-lab)

## Algorithm and Model used
### 1. Visial-Language-Model based on the transformers architecture 
- #### We settled on a VLM approach to address this problem.
- #### This type of model integrates visual and textual information using deep-learning algorithms
- #### It's based on the transformers architecture, enabling it to understand and relate images with languages
### 2. Selected model: moondream
- #### Chosen for it's compact size, powerful performance and fast inference.

[🔝 Back to Top](#table-of-contents-for-the-first-lab)
