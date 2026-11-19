Image Heatmap Overlay

A lightweight Python script that takes an image and an anomaly score map (z), then blends a smooth color heatmap (blue → cyan → green → yellow → red) over the hotspots. 

Setup

Install the required packages: 

numpy scipy pillow

Quick Start

1. Drop your test image into the project folder and name it input.jpg (or change input_path inside heatmap.py).
2. Run the script:

The script will save the result as heatmap_output.png and open it automatically. and if you want it in BMP of tiff can change it in code 

You can tweak how the heatmap looks inside heat_view(): 

red_at
(0 to 12): Sets the score threshold where pixels turn solid red. Keeps color scaling consistent across different images.
radius
(0 to 8): Controls how wide the soft glow spreads around tiny defect spots so they are easier to see.
mask
A boolean array matching the image size. Set pixels to False if you want to ignore background areas.
