
# Olympic Boxing Punch Classification

This project classifies boxing punches from videos using a combination of YOLO (for object detection) and LSTM (for punch classification).

## Setup Instructions

### Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### Install Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

Or manually install the dependencies:

```bash
pip install opencv-python torch torchvision scikit-learn ultralytics
```

## Data Preparation
The dataset can be downloaded from - https://www.kaggle.com/datasets/piotrstefaskiue/olympic-boxing-punch-classification-video-dataset/data
or can be incorporated in the code by using - 
```bash
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "piotrstefaskiue/olympic-boxing-punch-classification-video-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())
```


### Processing the Dataset

The script processes the dataset by extracting bounding boxes and frames from the videos. The processed data is cached in a `.pkl` file for efficient reloading.

## Training the Model

To train the model, run:

```bash
python detect.py
```

This will:
- Load the dataset and preprocess it.
- Train the LSTM model.
- Save the trained model as `lstm_model.pth`.

### Training Hyperparameters
- **Learning Rate**: 0.0005
- **Batch Size**: 2
- **Epochs**: 20
- **Optimizer**: Adam (with weight decay of 1e-4)

## Model Save/Load

The model is saved after training using `torch.save()`:

```python
# Load the trained model
model = PunchLSTM()  # Initialize the model architecture first
model.load_state_dict(torch.load("lstm_model.pth"))  # Load saved weights
model.eval()  # Set to evaluation mode
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

This is the full `README.md` content you can directly paste into your `README.md` file in your GitHub repository. Let me know if you need any more modifications!
