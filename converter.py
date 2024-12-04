import numpy as np
import pdf2image
import cv2
import matplotlib.pyplot as plt

# def convert_pdf_to_image(document, dpi):
#     images = []
#     images.extend(
#                     list(
#                         map(
#                             lambda image: cv2.cvtColor(
#                                 np.asarray(image), code=cv2.COLOR_RGB2BGR
#                             ),
#                             pdf2image.convert_from_path(document, dpi=dpi,poppler_path=r'C:\Users\engte\Materiais de Estudo\Faturas\leitor-de-faturas\poppler-24.08.0\Library\bin'),
#                         )
#                     )
#                 )
#     return images


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


pages = convert_pdf_to_image('arquivos/test_2.pdf', 300)
# Load the template
template = cv2.imread("template_2.png")
# Process the first page
first_page = pages[0]

# Find and enlarge the field
enlarged_field, location = find_and_enlarge_field(first_page, template, scale=2.0, threshold=0.4)

# Display the result if a match is found
if enlarged_field is not None:
    plt.imshow(cv2.cvtColor(enlarged_field, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Detected and Enlarged Field")
    plt.show()

    # Optionally save the enlarged field
    cv2.imwrite('outputs/enlarged_field.png', enlarged_field)
else:
    print("Field not detected.")



# images = convert_pdf_to_image('test.pdf', 400)
# # Number of pages in the pdf
# print(len(images))
# print(images[0].shape)

# first_page = images[0] # Let's work with the first page of the pdf
# plt.imshow(first_page)
# plt.show()