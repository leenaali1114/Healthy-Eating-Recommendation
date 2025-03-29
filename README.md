# Personalized Healthy Eating Suggestions

This project is a Flask-based web application that analyzes images of fruits and vegetables to determine their freshness and provides personalized recipe suggestions based on the analyzed ingredients and other available ingredients input by the user.

## Project Overview

The application uses a deep learning model to classify whether a fruit or vegetable is fresh or rotten. Based on the classification and other ingredients provided by the user, the system recommends healthy recipes.

### Features

- Upload and analyze images of fruits and vegetables
- Detect freshness status of produce using a trained machine learning model
- Identify the type of fruit/vegetable in the image
- Input additional available ingredients
- Receive personalized recipe recommendations based on the freshness status and available ingredients
- Optimize food usage by suggesting appropriate recipes for less-fresh produce

## Implementation Details

### Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: NumPy, Pillow

### Project Structure

```
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── model/                  # Trained ML model directory
│   └── model.h5            # Trained model file
├── static/                 # Static assets
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript files
│   └── uploads/            # Uploaded images
├── templates/              # HTML templates
│   ├── index.html          # Main page template
│   └── result.html         # Results page template
└── utils/                  # Utility modules
    ├── freshness.py        # Freshness detection module
    └── recipes.py          # Recipe recommendation module
```

## Setup and Running Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the model file:
   - The model file (model.h5) can be downloaded from [this link](https://drive.google.com/drive/folders/1hL7vvRZ9jHW9fpYeKBmlc-u5uUi44H-n?usp=sharing)
   - Place the model file in the `model/` directory

### Running the Application

1. Make sure the model file is in the correct location (`model/model.h5`)

2. Start the Flask server:
   ```
   python app.py
   ```

3. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Dataset Information

The model was trained on the Fruits and Vegetables Image Recognition Dataset. The dataset includes images of fresh and rotten fruits and vegetables for classification purposes.

### Dataset Download

The dataset and project resources can be accessed from:
- [Fruits and Vegetables Classification Dataset](https://www.kaggle.com/datasets/moltean/fruits)
- [Project Resources & Model Files](https://drive.google.com/drive/folders/1hL7vvRZ9jHW9fpYeKBmlc-u5uUi44H-n?usp=sharing)

## Important Documents

Additional project documents and resources are available at:
- [Project Documentation and Resources](https://drive.google.com/drive/folders/1hL7vvRZ9jHW9fpYeKBmlc-u5uUi44H-n?usp=sharing)

## Usage

1. On the homepage, upload an image of a fruit or vegetable
2. Enter any additional ingredients you have available (comma-separated)
3. Submit the form to get analysis results
4. View the freshness analysis and recipe recommendations tailored to your ingredients

## Extending the Project

To extend this project, you can:
- Add more sophisticated image recognition models
- Expand the recipe database
- Implement user accounts to save favorite recipes
- Integrate with grocery shopping APIs
- Add nutritional information to recipes

## License

This project is licensed under the MIT License - see the LICENSE file for details. 