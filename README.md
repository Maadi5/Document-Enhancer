# Document-Enhancer
Given an image of a document captured under real world conditions (eg: by a smartphone camera), this work aims to enhance the contents of said document. This is done by offsetting the shadows/lighting irregularities on the sheet as well as enhancing the contrast of the image to highlight the details.
This is designed to work on documents of any kind, that have a combination of text and images.

## Prerequisites:

1. Numpy
2. Matplotlib
3. OpenCV

## How to run:
1. Clone the repository
2. Load input images into the Input directory (by default: 'inputs' folder in repo)
3. For all defaults, run the command: `python doc_enhancer.py`
4. All the enhanced images will be saved in the output folder (by default: 'outputs' folder in repo)
  
## Additional Options:
1. To change the contrast level, pass argument `--contrast_val` along with your desired integer value. (default is 3)
2. To change the input directory, pass argument `--inputs_dir` along with your desired path
3. To change the output directory, pass argument `--outputs_dir` along with your desired path 
### Example:
`python doc_enhancer.py --inputs_dir ./inps  --outputs_dir  ./outps  --contrast_val  2`
