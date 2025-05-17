import os
from ultralytics import YOLO
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def get_unique_save_dir(base_dir):
    """Generate a unique save directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"train_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def train_model():
    # Configuration
    yaml_path = '/home/aman-nvidia/My_files/ai_projects/anpr_django_system/anpr_dataset/plates.yaml'
    base_save_dir = 'anpr_django_system/runs/plate_detection/train'  # Base directory

    # Get the most recent checkpoint if exists
    previous_runs = sorted([d for d in os.listdir(base_save_dir) if d.startswith('train_')], reverse=True)
    last_checkpoint = None
    if previous_runs:
        last_checkpoint_path = os.path.join(base_save_dir, previous_runs[0], 'weights', 'last.pt')
        if os.path.exists(last_checkpoint_path):
            last_checkpoint = last_checkpoint_path
            logging.info(f"Found previous checkpoint: {last_checkpoint}")

    # Model initialization with resume logic
    try:
        if last_checkpoint:
            logging.info(f"Resuming training from: {last_checkpoint}")
            model = YOLO(last_checkpoint)
            resume = True
            save_dir = os.path.dirname(os.path.dirname(last_checkpoint))  # Use existing run directory
        else:
            logging.info("Starting new training from pretrained weights")
            model = YOLO("yolov8n.pt")
            resume = False
            save_dir = get_unique_save_dir(base_save_dir)  # Create new run directory

        # Training configuration
        results = model.train(
            data=yaml_path,
            epochs=10,
            imgsz=640,
            batch=2,
            resume=resume,
            project=os.path.dirname(save_dir),  # Project is the base directory
            name=os.path.basename(save_dir),   # Name is the unique run directory
            save=True,
            save_period=1,  # Save checkpoint every epoch
            exist_ok=True,  # Allow overwriting in the current run directory if resuming
            device='0'  # Use GPU 0 explicitly
        )

        # Post-training info
        logging.info(f"Training completed. Results saved to: {save_dir}")
        best_checkpoint = os.path.join(save_dir, 'weights', 'best.pt')
        if os.path.exists(best_checkpoint):
            logging.info(f"Best model saved to: {best_checkpoint}")

        return results

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        if 'last_checkpoint' in locals() and os.path.exists(last_checkpoint):
            logging.info(f"You can resume later by running this script again.")
        raise

if __name__ == "__main__":
    train_model()