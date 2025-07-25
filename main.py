import re
import pytesseract
import cv2
import numpy as np
import argparse

class SimpleAadhaarExtractor:
    def __init__(self, debug=False):
        self.debug = debug
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
                cv2.imwrite("debug_preprocessed_final.jpg", dilated)
                print("Saved preprocessed image as 'debug_preprocessed_final.jpg'")
                
            return dilated
        except Exception as e:
            print(f"Error in final preprocessing: {e}")
            return None

    def extract_text(self, processed_img):
        """Extract text from preprocessed image"""
        if processed_img is None:
            return ""
        
        # Use PSM 6 for better text segmentation
        config = '--psm 6 --oem 3'
        text = pytesseract.image_to_string(processed_img, config=config)
        
        if self.debug:
            print("\nRaw Extracted Text:")
            print("="*40)
            print(text)
            print("="*40)
            
        return text

    def extract_info(self, text):
        """Extract key fields with improved logic"""
        info = {
            "name": self._extract_name(text),
            "aadhaar_no": self._extract_aadhaar(text),
            "dob": self._extract_dob(text),
            "gender": self._extract_gender(text),
            "formatted_output": ""
        }
        
        # Format the output
        formatted = []
        if info["name"]: formatted.append(f"Name: {info['name']}")
        if info["aadhaar_no"]: formatted.append(f"Aadhaar Number: {info['aadhaar_no']}")
        if info["dob"]: formatted.append(f"Date of Birth: {info['dob']}")
        if info["gender"]: formatted.append(f"Gender: {info['gender']}")
        
        info["formatted_output"] = "\n".join(formatted) if formatted else "No information extracted"
        return info

    def _extract_name(self, text):
        """Enhanced name extraction with improved artifact and non-name pattern removal"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        name_candidates = []
        
        # Common OCR artifacts and non-name words to exclude
        ocr_artifacts = {
            'va', 'yo', 'gr', 'es', 'cg', 'socio', 'ait', 'cearey', 'att', 'gan', 'sem', 'ae', 'tar', 'ph',
            'far', 'osh', 'pa', 'eee', 'sree', 'btu', 'sta', 'hight', 'ar', 'hrw', 'nd', 'lal', 'qeaxey',
            'wrote', 'ieg', 'ly', 'al', 'nee', 'soy', 'ie', 'arsh', 'swe', 'bay', 'arr', 'sure', 'onlndia'
        }
        
        # Non-name patterns (case-insensitive)
        non_name_patterns = [
            r'government\s*of\s*india',
            r'governmentof\s*india',
            r'governmentofindia',
            r'india',
            r'uid',
            r'aadhaar',
            r'आधार',
            r'यूआईडी',
            r'dob',
            r'gender',
            r'male',
            r'female'
        ]
        
        for line in lines:
            # Skip lines with numbers or known non-name patterns
            if re.search(r'\d{4,}', line):  # Match 4 or more digits
                continue
                
            # Check for non-name patterns (case-insensitive)
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in non_name_patterns):
                continue
            
            # Clean the line: remove special characters but keep spaces and apostrophes (for names like O'Connor)
            cleaned = re.sub(r"[^A-Za-z'\s]", '', line).strip()
            # Split into words
            parts = cleaned.split()
            
            # Filter out:
            # 1. Short words (<=2 chars)
            # 2. Known artifacts
            # 3. Words that are all uppercase (likely headers/acronyms)
            valid_parts = [
                part for part in parts 
                if (len(part) > 2 and 
                    part.lower() not in ocr_artifacts and
                    not part.isupper())
            ]
            
            # Require at least 2 valid parts for a name
            if len(valid_parts) >= 2:
                name = ' '.join(valid_parts)
                if name:
                    name_candidates.append(name)
        
        if not name_candidates:
            return None
            
        # Return the longest candidate (likely the full name)
        # Also filter out candidates that are too short (less than 5 characters)
        valid_candidates = [name for name in name_candidates if len(name) >= 5]
        return max(valid_candidates, key=len, default=None)

    def _extract_aadhaar(self, text):
        """Robust Aadhaar number extraction with improved pattern matching and validation"""
        replacements = {
            'O': '0', 'o': '0', 'Q': '0',
            'l': '1', 'I': '1', '|': '1',
            'S': '5', 'Z': '2', 'B': '8',
            ' ': '', '-': '', '.': '', '/': '', '\\': ''
        }
        
        # Apply replacements
        cleaned_text = text
        for old, new in replacements.items():
            cleaned_text = cleaned_text.replace(old, new)
        
        # Look for Aadhaar numbers in various formats
        patterns = [
            r'(?:aadhaar|uid|आधार|यूआईडी|number|no\.?)[^\d]*(\d{4}[-\s]?\d{4}[-\s]?\d{4})',
            r'\b(\d{4}[-\s]?\d{4}[-\s]?\d{4})\b',
            r'(?<!\d)\d{12}(?!\d)',
            r'(\d{4})\s*\n\s*(\d{8})',
            r'(\d[\dOoQlISZB]{3}[\dOoQlISZB\-\.\s]{4}[\dOoQlISZB]{4})'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                num_str = ''.join([g for g in match.groups() if g])
                clean_num = re.sub(r'[^\d]', '', num_str)
                if self._validate_aadhaar(clean_num):
                    return f"{clean_num[:4]} {clean_num[4:8]} {clean_num[8:12]}"
        
        # Special case: Look for numbers near "Government of India"
        govt_lines = [i for i, line in enumerate(text.split('\n')) 
                    if 'government of india' in line.lower()]
        
        if govt_lines:
            for i in govt_lines:
                next_lines = text.split('\n')[i+1:i+4]
                for line in next_lines:
                    nums = re.findall(r'\d{4}\s?\d{4}\s?\d{4}', line)
                    for num in nums:
                        clean_num = re.sub(r'[^\d]', '', num)
                        if self._validate_aadhaar(clean_num):
                            return f"{clean_num[:4]} {clean_num[4:8]} {clean_num[8:12]}"
        
        return None

    def _validate_aadhaar(self, number):
        """Validate Aadhaar number with multiple checks"""
        if not number or len(number) != 12 or not number.isdigit():
            return False
        
        invalid_patterns = [
            r'^(\d)\1{11}$',
            r'^1234.*',
            r'^(\d{4})\1\1$',
            r'^[0]{4}.*',
            r'^1{10}.*',
            r'^(\d)\1(\d)\2(\d)\3$'
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, number):
                return False
        
        return True

    def _extract_dob(self, text):
        """Extract date of birth with validation"""
        match = re.search(r'(?:DOB|Date of Birth)[^\d]*(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
        if not match:
            match = re.search(r'\b(\d{2}/\d{2}/\d{4})\b', text)
        if match:
            dob = match.group(1)
            try:
                day, month, year = map(int, dob.split('/'))
                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2025:
                    return dob
            except ValueError:
                pass
        return None

    def _extract_gender(self, text):
        """Robust gender extraction with enhanced OCR error handling"""
        # Normalize text: remove special chars but preserve slashes and spaces
        normalized_text = re.sub(r'[^a-zA-Z0-9/\s]', ' ', text.lower())
        
        # Common OCR misreads for gender terms
        ocr_corrections = {
            'make': 'male',
            'maie': 'male',
            'ma1e': 'male',
            'ma|e': 'male',
            'femaie': 'female',
            'fema1e': 'female',
            'feme': 'female',
            'fe make': 'female',  # Common two-word misread
            'mie': 'male',       # From your example "Mie"
            'fe': 'female'       # From your example "fe"
        }
        
        # Apply OCR corrections
        for wrong, correct in ocr_corrections.items():
            normalized_text = normalized_text.replace(wrong, correct)
        
        # Gender patterns in priority order
        patterns = [
            # 1. Explicit gender markers near keywords (most reliable)
            r'(?:gender|sex|लिंग|लिङ्ग)[^\w\n]*(male|female|m|f|महिला|पुरुष|स्त्री|पुरूष)',
            r'(?:dob|date of birth|जन्म तिथि)[^\n]*(male|female|m|f|महिला|पुरुष|स्त्री|पुरूष)',
            
            # 2. Standalone gender markers
            r'(?<!\w)(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री)(?!\w)',
            
            # 3. M/F near slash (common in forms)
            r'(?<!\w)(m|f)(?:\s*[/\\]\s*(m|f))?(?!\w)',
            
            # 4. Common OCR patterns seen in Aadhaar cards
            r'\b(m[ai][l1e]e|f[ae]m[ai][l1e]e)\b'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                # Get the first non-None group
                gender = next((g for g in match.groups() if g), None)
                if gender:
                    gender = gender.lower()
                    if gender in ['male', 'm', 'पुरुष', 'पुरूष']:
                        return 'Male'
                    elif gender in ['female', 'f', 'महिला', 'स्त्री']:
                        return 'Female'
        
        # Special handling for your specific case where "Mie | fe Make" appears
        if 'mie' in normalized_text or 'fe make' in normalized_text:
            # Check which term appears first
            male_pos = normalized_text.find('mie')
            female_pos = normalized_text.find('fe make')
            
            if male_pos != -1 and (female_pos == -1 or male_pos < female_pos):
                return 'Male'
            elif female_pos != -1:
                return 'Female'
        
        # Final fallback: count occurrences
        male_terms = sum(normalized_text.count(term) for term in ['male', 'm', 'पुरुष', 'पुरूष'])
        female_terms = sum(normalized_text.count(term) for term in ['female', 'f', 'महिला', 'स्त्री'])
        
        if male_terms > female_terms:
            return 'Male'
        elif female_terms > male_terms:
            return 'Female'
        
        return None

    def process(self, img_path):
        """Process image with multiple preprocessing attempts if needed"""
        try:
            # First attempt with original preprocessing
            processed_img = self.preprocess_image(img_path)
            text = self.extract_text(processed_img)
            info = self.extract_info(text)
            
            # Check if all four fields are extracted
            extracted_fields = sum(1 for value in [info["name"], info["aadhaar_no"], info["dob"], info["gender"]] if value)
            if extracted_fields == 4:
                if self.debug:
                    print(f"Extracted all 4 fields with original preprocessing.")
                return info
            
            # If fewer than 4 fields extracted, try alternative preprocessing
            if self.debug:
                print(f"Extracted {extracted_fields}/4 fields with original preprocessing. Trying alternative preprocessing...")
            
            processed_img_alt = self.preprocess_image_alternative(img_path)
            text_alt = self.extract_text(processed_img_alt)
            info_alt = self.extract_info(text_alt)
            
            # Combine results, prioritizing non-None values
            for key in ["name", "aadhaar_no", "dob", "gender"]:
                if not info[key] and info_alt[key]:
                    info[key] = info_alt[key]
            
            # Check again if all fields are extracted
            extracted_fields = sum(1 for value in [info["name"], info["aadhaar_no"], info["dob"], info["gender"]] if value)
            if extracted_fields == 4:
                if self.debug:
                    print(f"Extracted all 4 fields after alternative preprocessing.")
                # Update formatted output
                formatted = []
                if info["name"]: formatted.append(f"Name: {info['name']}")
                if info["aadhaar_no"]: formatted.append(f"Aadhaar Number: {info['aadhaar_no']}")
                if info["dob"]: formatted.append(f"Date of Birth: {info['dob']}")
                if info["gender"]: formatted.append(f"Gender: {info['gender']}")
                info["formatted_output"] = "\n".join(formatted) if formatted else "No information extracted"
                return info
            
            # If still fewer than 4 fields, try final preprocessing
            if self.debug:
                print(f"Extracted {extracted_fields}/4 fields after alternative preprocessing. Trying final preprocessing...")
            
            processed_img_final = self.preprocess_image_final(img_path)
            text_final = self.extract_text(processed_img_final)
            info_final = self.extract_info(text_final)
            
            # Combine results, prioritizing non-None values
            for key in ["name", "aadhaar_no", "dob", "gender"]:
                if not info[key] and info_final[key]:
                    info[key] = info_final[key]
            
            # Update formatted output
            formatted = []
            if info["name"]: formatted.append(f"Name: {info['name']}")
            if info["aadhaar_no"]: formatted.append(f"Aadhaar Number: {info['aadhaar_no']}")
            if info["dob"]: formatted.append(f"Date of Birth: {info['dob']}")
            if info["gender"]: formatted.append(f"Gender: {info['gender']}")
            
            info["formatted_output"] = "\n".join(formatted) if formatted else "No information extracted"
            
            if self.debug:
                extracted_fields_final = sum(1 for value in [info["name"], info["aadhaar_no"], info["dob"], info["gender"]] if value)
                print(f"Final extracted fields: {extracted_fields_final}/4")
            
            return info
        except Exception as e:
            print(f"Error processing image: {e}")
            return {"formatted_output": "Error processing image"}

def main():
    parser = argparse.ArgumentParser(description="Simple Aadhaar Info Extractor")
    parser.add_argument("file_path", help="Path to Aadhaar card image")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    extractor = SimpleAadhaarExtractor(debug=args.debug)
    result = extractor.process(args.file_path)
    
    print("\nExtracted Information:")
    print("="*40)
    print(result["formatted_output"])
    print("="*40)

if __name__ == "__main__":
    main()