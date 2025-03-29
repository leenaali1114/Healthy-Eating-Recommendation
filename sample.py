import os
import sys
from PIL import Image
from utils.freshness import load_freshness_model, predict_freshness

def test_single_image(image_path):
    """
    Test the freshness detection model on a single image.
    
    Args:
        image_path: Path to the image file
    """
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Load the model
    print("Loading model...")
    model = load_freshness_model()
    
    # Process the image
    print(f"Processing image: {image_path}")
    
    try:
        # Display the image if possible
        try:
            img = Image.open(image_path)
            img.show()  # This will open the image in the default image viewer
        except Exception as e:
            print(f"Could not display image: {e}")
        
        # Predict freshness
        result = predict_freshness(image_path, model)
        
        # Print detailed results
        print("\n--- Prediction Results ---")
        freshness = "Fresh" if result['is_fresh'] else "Rotten"
        print(f"Prediction: {freshness}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Fruit Type: {result['fruit_type']}")
        
        # Print raw prediction value (this comes from the debug print in predict_freshness)
        print("\nCheck the raw prediction value above to see if it's always close to 0 or 1.")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    # Check if an image path was provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_single_image(image_path)
    else:
        # If no image path was provided, ask for one
        image_path = input("dataset/Test/freshapples/a_f001.png ")
        test_single_image(image_path)