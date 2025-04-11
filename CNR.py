#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[8]:


import cv2
import pytesseract


# In[9]:


img = cv2.imread('car.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(gray)
print("Detected Text:", text)


# In[11]:


import cv2
import pytesseract
from PIL import Image

# Set tesseract executable path (Windows only)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# In[ ]:


# Load the image
image_path = 'car.jpg'  # Change this to your image file
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Haar cascade for license plates
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Detect number plates
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Loop through detections
for (x, y, w, h) in plates:
    # Draw rectangle around detected plate
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Crop the plate region
    plate_img = gray[y:y+h, x:x+w]

    # Optional: apply some preprocessing for better OCR
    _, plate_thresh = cv2.threshold(plate_img, 127, 255, cv2.THRESH_BINARY)

    # Save or process the cropped plate image
    cv2.imwrite('cropped_plate.jpg', plate_thresh)

    # OCR using pytesseract
    plate_text = pytesseract.image_to_string(plate_thresh, config='--psm 8')  # 8 = single word/line
    print("Detected Number Plate Text:", plate_text.strip())

# Show image with detected plate
cv2.imshow("Number Plate Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




