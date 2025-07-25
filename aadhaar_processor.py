    def preprocess_image(self, img_path):
        """Enhanced image preprocessing with zoom (original method)"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # Zoom (upscale 2x) to improve text readability
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive thresholding for better contrast
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Optional: Dilate to connect text regions
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            if self.debug:
                cv2.imwrite("debug_preprocessed_original.jpg", dilated)
                print("Saved preprocessed image as 'debug_preprocessed_original.jpg'")
                
            return dilated
        except Exception as e:
            print(f"Error in original preprocessing: {e}")
            return None

    def preprocess_image_alternative(self, img_path):
        """Alternative image preprocessing with zoom"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # Zoom (upscale 2x) to improve text readability
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising to reduce OCR noise
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean text
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            if self.debug:
                cv2.imwrite("debug_preprocessed_alternative.jpg", processed)
                print("Saved preprocessed image as 'debug_preprocessed_alternative.jpg'")
                
            return processed
        except Exception as e:
            print(f"Error in alternative preprocessing: {e}")
            return None

-------------------------------------------------------------------------------------------
def preprocess_image(self, img_path):
    """Enhanced image preprocessing (original method)"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding for better contrast
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Optional: Dilate to connect text regions
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        if self.debug:
            cv2.imwrite("debug_preprocessed_original.jpg", dilated)
            print("Saved preprocessed image as 'debug_preprocessed_original.jpg'")
            
        return dilated
    except Exception as e:
        print(f"Error in original preprocessing: {e}")
        return None

def preprocess_image_alternative(self, img_path):
    """Alternative image preprocessing with zoom and denoising"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Zoom (upscale 2x) to improve text readability
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising to reduce OCR noise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean text
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        if self.debug:
            cv2.imwrite("debug_preprocessed_alternative.jpg", processed)
            print("Saved preprocessed image as 'debug_preprocessed_alternative.jpg'")
            
        return processed
    except Exception as e:
        print(f"Error in alternative preprocessing: {e}")
        return None

def preprocess_image_final(self, img_path):
    """Final attempt preprocessing focusing on numbers"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(filtered, -1, kernel)
        
        # Thresholding with OTSU
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (black text on white background)
        thresh = cv2.bitwise_not(thresh)
        
        if self.debug:
            cv2.imwrite("debug_preprocessed_final.jpg", thresh)
            print("Saved preprocessed image as 'debug_preprocessed_final.jpg'")
            
        return thresh
    except Exception as e:
        print(f"Error in final preprocessing: {e}")
        return None

def extract_information(self, img_path):
    """Main extraction logic with three-stage processing"""
    # First try with original preprocessing
    preprocessed = self.preprocess_image(img_path)
    text = self.extract_text(preprocessed)
    extracted = self.parse_text(text)
    
    if len(extracted) >= 4:  # All fields found
        return extracted
    
    # Second try with alternative preprocessing
    if len(extracted) < 4:
        print(f"Extracted {len(extracted)}/4 fields with original preprocessing. Trying alternative preprocessing...")
        preprocessed = self.preprocess_image_alternative(img_path)
        text = self.extract_text(preprocessed)
        extracted = self.parse_text(text)
    
    # Final try with number-focused preprocessing if still missing fields
    if len(extracted) < 4:
        print(f"Extracted {len(extracted)}/4 fields with alternative preprocessing. Trying final number-focused preprocessing...")
        preprocessed = self.preprocess_image_final(img_path)
        text = self.extract_text(preprocessed)
        
        # Special parsing for numbers (Aadhaar and DOB)
        numbers = re.findall(r'\d{2,}', text)
        aadhaar_candidates = [n for n in numbers if len(n) in [8, 12]]
        dob_candidates = [n for n in numbers if len(n) in [6, 8] and '/' not in n]
        
        # Update extracted fields
        if 'aadhaar_number' not in extracted and aadhaar_candidates:
            extracted['aadhaar_number'] = aadhaar_candidates[0]
        if 'dob' not in extracted and dob_candidates:
            # Format DOB if found
            dob_str = dob_candidates[0]
            if len(dob_str) == 8:
                extracted['dob'] = f"{dob_str[:2]}/{dob_str[2:4]}/{dob_str[4:]}"
            elif len(dob_str) == 6:
                extracted['dob'] = f"{dob_str[:2]}/{dob_str[2:4]}/19{dob_str[4:]}"
    
    print(f"Final extracted fields: {len(extracted)}/4")
    return extracted