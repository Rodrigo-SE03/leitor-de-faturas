import numpy as np
import pdf2image
import cv2

def convert_pdf_to_image(document, dpi):
    images = [
        cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR)
        for image in pdf2image.convert_from_path(document, dpi=dpi,poppler_path=r'C:\Users\engte\Materiais de Estudo\Faturas\leitor-de-faturas\poppler-24.08.0\Library\bin')
    ]
    return images

def find_and_enlarge_field(image, template, scale=2.0, threshold=0.4):
    """
    Find a field in the image using template matching and enlarge it.
    """
    # Convert images to grayscale for matching
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Match template
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:  # If a match is found above the threshold
        x, y = max_loc
        h, w = gray_template.shape
        
        # Crop the matched region
        cropped_field = image[y:y+h, x:x+w]
        
        # Enlarge the cropped field
        enlarged_field = cv2.resize(
            cropped_field, 
            (int(w * scale), int(h * scale)), 
            interpolation=cv2.INTER_CUBIC
        )
        
        return enlarged_field, (x, y, w, h)
    else:
        print("No matching field found.")
        return None, None


def capture_images(pdf_path, templates_folder_path):
    pages = convert_pdf_to_image(pdf_path, 300)

    for i in range(3):
        template = cv2.imread(f"{templates_folder_path}/template_{i+1}.png")
        for page in pages:
            enlarged_field, location = find_and_enlarge_field(page, template, scale=2.0, threshold=0.4)
            if enlarged_field is not None:
                break
        if enlarged_field is not None:
            cv2.imwrite(f'outputs/field_{i+1}.png', enlarged_field)