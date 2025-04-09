# High Entropy Alloys
### An analysis of High entropy alloys using PCA, K-prototypes and ANNs

## Repo Structure
The repo is stuctured by seperating the data, figures and the source code into different folders. The source code can be found under the "src" folder and contains the various scripts used for this project.
The data folder contains both the original and parsed versions of our dataset. The figures folder contains various figures generated throughout the analysis seperated into 3 folders. "PCA" corresponds to the
figures related to the PCA analysis we conducted. "ANN" corresponds to the figures related to the Large ANN. "ANNs" corresponds to the figures related to the Multi-layered ANN.

## Source code
The source directory contains all of our different scripts related to the project. To run one, you should make sure to have all the required dependencies installed. If one is not installed you can install it
using pip with the following example for installing matplotlib.

  pip3 install matplotlib

To run one of the examples, run it through python using the filename as the arguement. For example,

  python3 ANN.py

will run the ANN model training. The directory also contains some tools such as PCA_tools.py and general_tools.py. These files contain various helpful functions we created and may have accessed in other files. 
- The PCA.py file contains the PCA analysis script we completed on our dataset
- The k_prototypes.py file contains the implementation of our K-prototypes analysis
- The ANN.py file contains the implementation of our Large ANN model and analysis of it
- The Multilevel_ANN.py file contains the implementation of our Multilevel ANN model and analysis of it
- The parse.py file contains a parsing script that generated the parsed csv
