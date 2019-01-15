# Geomapper - DSI Capstone
Galvanize G83 Data Science Immersive Phoenix -- Capstone


This repo contains code in support of my Galvanize Data Science Immersive capstone project.
The project uses multispectral satellite imagery of the state of Arizona acquired by ESA's Sentinel-2 mission. Using a "Geology" band combination of 2-4-12 and training labels consisting of the State's Geologic map units trained a neural network architecture model backend of TensorFlow DeepLab's Xception_65 using the Rastervision library to construct the experiment. The project attempts to predict the geology using semantic segmentation, or pixel-wise classification, from satellite imagery.