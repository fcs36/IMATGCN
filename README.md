
##File Descriptions

- data_load.py: Loads time series data stored in the `data/` folder.
- data_process.py: Preprocesses the time series data and transforms it into graph data suitable for GCN models.
- MA_TGCN.py: Contains the implementation of the Multi-head Attention Temporal Graph Convolutional Network model.
- main.py: Runs the training and evaluation procedures for the forecasting model.
- Evaluation_Metrics.py: Defines functions to compute evaluation metrics.
- data/: Directory containing the raw input time series data.
- result/: Directory where prediction results are saved.

##How to Run

1. Make sure you have all dependencies installed
2. Place your time series data in the `data/` folder.
3. Run the training and testing script:

python main.py

