# Enhanced utility module for license plate recognition system

# Import required dependencies
import numpy as np
import string
from typing import Dict, List, Tuple, Union
import easyocr

# Constants
CHAR_TO_INT = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
INT_TO_CHAR = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}
LICENSE_LENGTH = 7
POSITION_MAPPING = {
    0: INT_TO_CHAR, 1: INT_TO_CHAR, 4: INT_TO_CHAR, 
    5: INT_TO_CHAR, 6: INT_TO_CHAR, 2: CHAR_TO_INT, 3: CHAR_TO_INT
}

# Initialize the OCR reader (lazy loading for better performance)
_reader = None

def get_ocr_reader() -> easyocr.Reader:
    """Initialize and return the OCR reader singleton."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=True)
    return _reader

def get_vehicle(license_plate: List[float], vehicle_track_ids: List[List[float]]) -> Tuple[float, ...]:
    """
    Associate a license plate with a vehicle using bounding box containment.
    
    Args:
        license_plate: [x1, y1, x2, y2, score, class_id] of detected license plate
        vehicle_track_ids: List of vehicle bounding boxes with IDs [x1, y1, x2, y2, id]
        
    Returns:
        Tuple: (x1, y1, x2, y2, car_id) of containing vehicle or (-1, -1, -1, -1, -1) if not found
    """
    x1, y1, x2, y2, _, _ = license_plate
    
    for vehicle in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle
        if (x1 > xcar1 and y1 > ycar1 and 
            x2 < xcar2 and y2 < ycar2):
            return vehicle
    
    return (-1, -1, -1, -1, -1)

def read_license_no(croped_license_plate: np.ndarray) -> Tuple[Union[str, int], float]:
    """
    Extract and validate license plate number from cropped image.
    
    Args:
        croped_license_plate: Cropped license plate image
        
    Returns:
        Tuple: (formatted_license_text, confidence_score) or (0, 0) if invalid
    """
    detections = get_ocr_reader().readtext(croped_license_plate)
    
    for bbox, text, score in detections:
        clean_text = text.upper().replace(' ', '')
        if license_complies_format(clean_text):
            return format_license(clean_text), score
    
    return 0, 0

def write_csv(hold_results: Dict, output_path: str) -> None:
    """
    Write detection results to CSV file with standardized format.
    
    Args:
        hold_results: Nested dictionary containing all detection data
        output_path: Path to output CSV file
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write('num_frames,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')
        
        # Write data rows
        for frame_num, frame_data in hold_results.items():
            for car_id, car_data in frame_data.items():
                if not all(k in car_data for k in ('car', 'license_plate')):
                    continue
                    
                car_bbox = car_data['car']['bbox']
                lp_data = car_data['license_plate']
                
                if 'text' not in lp_data:
                    continue
                
                # Format bounding boxes as strings
                car_bbox_str = f"[{' '.join(map(str, car_bbox))}]"
                lp_bbox_str = f"[{' '.join(map(str, lp_data['bbox']))}]"
                
                # Write CSV row
                f.write(f"{frame_num},{car_id},{car_bbox_str},{lp_bbox_str},"
                       f"{lp_data['bbox_score']},{lp_data['text']},{lp_data['text_score']}\n")

def license_complies_format(text: str) -> bool:
    """
    Validate license plate format according to predefined rules.
    
    Args:
        text: Extracted license plate text
        
    Returns:
        bool: True if text complies with required format
    """
    if len(text) != LICENSE_LENGTH:
        return False
    
    # Define validation rules for each character position
    position_rules = [
        (string.ascii_uppercase | INT_TO_CHAR.keys()),  # Position 0
        (string.ascii_uppercase | INT_TO_CHAR.keys()),  # Position 1
        (set('0123456789') | CHAR_TO_INT.keys()),       # Position 2
        (set('0123456789') | CHAR_TO_INT.keys()),       # Position 3
        (string.ascii_uppercase | INT_TO_CHAR.keys()),  # Position 4
        (string.ascii_uppercase | INT_TO_CHAR.keys()),  # Position 5
        (string.ascii_uppercase | INT_TO_CHAR.keys())   # Position 6
    ]
    
    return all(text[i] in position_rules[i] for i in range(LICENSE_LENGTH))

def format_license(text: str) -> str:
    """
    Format license plate text by converting characters according to mapping rules.
    
    Args:
        text: Raw license plate text
        
    Returns:
        str: Formatted license plate text
    """
    return ''.join(
        POSITION_MAPPING[i].get(char, char) 
        for i, char in enumerate(text[:LICENSE_LENGTH])
    )

# ====================================================== END =====================================================
# ======================================================     =====================================================