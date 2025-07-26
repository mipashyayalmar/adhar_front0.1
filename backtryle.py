import re
import pytesseract
import cv2
import numpy as np
import argparse

class AadhaarAddressExtractor:
    def __init__(self, debug=False):
        self.debug = debug
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.pin_code_pattern = r'\b\d{6}\b'

    def preprocess_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            if self.debug:
                cv2.imwrite("debug_preprocessed.jpg", processed)
                print("Saved preprocessed image as 'debug_preprocessed.jpg'")

            return processed
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    def extract_text(self, processed_img):
        if processed_img is None:
            return ""

        config = ('-l eng --psm 6 --oem 3 '
                 '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,/.- ()')
        text = pytesseract.image_to_string(processed_img, config=config)

        if self.debug:
            print("\nRaw Extracted Text:")
            print("="*40)
            print(text)
            print("="*40)

        return text

    def is_back_side(self, text):
        back_keywords = [
            r'unique\s*identification\s*authority\s*of\s*india',
            r'address',
            r'uidai',
            r'help@uidai\.gov\.in',
            r'www\.uidai\.gov\.in',
            r'1947',
            r'aadhaar'
        ]
        combined_text = ' '.join([line.strip() for line in text.split('\n') if line.strip()])
        match_count = sum(1 for kw in back_keywords if re.search(kw, combined_text, re.IGNORECASE))
        
        if self.debug:
            print(f"Back side detection match count: {match_count}")
            
        return match_count >= 3


    def clean_address_line(self, line):
        # Enhanced noise words and patterns
        noise_words = [
            r'\bIAddress\b', r'\bAddress[-:]\s*', r'\beee\b', r'\bsivas\b', 
            r'\.aa\b', r'\bes\b', r'\be\b', r'\bges\b', r'\bnre\b',
            r'\berasers\b', r'\bsats\b', r'\brees\b', r'\bspee\b',
            r'\ba\b', r'\bo\b', r'\bos\b', r'\bia\b', r'\bPENRASeohanaunee\b',
            r'\bees\b', r'\bee\b', r'\ -', r'\_', r'\bI\b',r'\bIAddress\b', r'\bIAddress[-:]\s*', r'\bIAddress\s*', r'\bIAddress\b'
        ]
        
        patterns_to_remove = [
            r'unique identification.*',
            r'uidai\.gov\.in.*',
            r'help@uidai\.gov\.in.*',
            r'1947.*',
            r'www\.uidai\.gov\.in.*',
            r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{2}:\d{2}',
            r'^\W+|\W+$'  # Remove leading/trailing non-word chars
        ] + noise_words

        # Initial aggressive cleaning
        cleaned = line
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
        # Standardize relationship indicators with proper spacing
        cleaned = re.sub(r'(\b[SsDdWwCc])/([Oo])\b', r' \1/O ', cleaned)
        
        # Split camelCase names but preserve all-caps words
        def split_name(match):
            word = match.group()
            if word.isupper() or '/' in word:
                return word
            return re.sub(r'([a-z])([A-Z])', r'\1 \2', word)
        
        cleaned = re.sub(r'\b[A-Za-z][a-z]*[A-Z][A-Za-z]*\b', split_name, cleaned)
        
        # Process address components
        words = []
        for word in cleaned.split():
            # Remove any remaining single character words except S/O components
            if len(word) == 1 and not word.isdigit() and word not in ['S', 'D', 'W', 'C', 'O']:
                continue
            
            # Handle hyphenated pincodes
            if re.match(r'^\d+-\d+$', word):
                word = word.replace('-', ' ')
            
            words.append(word)
        
        cleaned = ' '.join(words)
        
        # Final cleaning and formatting
        cleaned = re.sub(r'[^a-zA-Z0-9\s\-,./()]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Ensure proper spacing after commas
        cleaned = re.sub(r',(\S)', r', \1', cleaned)
        
        # Ensure consistent comma spacing and fix S/O cases
        cleaned = re.sub(r',(\S)', r', \1', cleaned)
        cleaned = re.sub(r'(\b[SsDdWwCc]/O)(\S)', r'\1 \2', cleaned)
        

        
        # Validate line
        if len(cleaned) < 10 or not re.search(r'[a-zA-Z]{3,}', cleaned):
            return None
            
        return cleaned



    def extract_address(self, text):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        address_lines = []
        found_start = False
        
        address_starters = [r'[Ss]/[Oo]', r'[Dd]/[Oo]', r'[Ww]/[Oo]', r'[Cc]/[Oo]']
        indian_states = [
            'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
            'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
            'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
            'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
            'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
            'Andaman and Nicobar', 'Chandigarh', 'Dadra and Nagar Haveli',
            'Daman and Diu', 'Delhi', 'Lakshadweep', 'Puducherry', 'Jammu and Kashmir',
            'Ladakh'
        ]
        exclude_patterns = [
            r'unique identification',
            r'uidai',
            r'help@uidai',
            r'www\.uidai',
            r'1947',
            r'\d{4}\s?\d{4}\s?\d{4}'
        ]
        
        for i, line in enumerate(lines):
            if any(re.search(pat, line, re.IGNORECASE) for pat in address_starters):
                found_start = True
                cleaned = self.clean_address_line(line)
                if cleaned:
                    address_lines.append(cleaned)
                
                for next_line in lines[i+1:]:
                    if any(re.search(pat, next_line, re.IGNORECASE) for pat in exclude_patterns):
                        break
                    cleaned_next = self.clean_address_line(next_line)
                    if cleaned_next:
                        address_lines.append(cleaned_next)
                        if (any(state in cleaned_next for state in indian_states) or
                            re.search(self.pin_code_pattern, cleaned_next)):
                            break
                break
        
        if address_lines:
            full_address = ', '.join(address_lines)
            full_address = re.sub(r'\s+', ' ', full_address).strip()
            full_address = re.sub(r',+', ',', full_address).strip(',')
            
            pin_match = re.search(self.pin_code_pattern, full_address)
            if pin_match:
                pin = pin_match.group(0)
                full_address = re.sub(self.pin_code_pattern, '', full_address).strip()
                full_address = re.sub(r'\s+', ' ', full_address).strip()
                full_address = re.sub(r',+', ',', full_address).strip(',')
                full_address = f"{full_address}, {pin}"
            
            if self.debug:
                print(f"Address lines considered: {address_lines}")
            
            return full_address if len(full_address) > 15 else None
        
        return None

    def process(self, img_path):
        try:
            processed_img = self.preprocess_image(img_path)
            if processed_img is None:
                return {"address": None, "formatted_output": "Error: Failed to process image"}
                
            text = self.extract_text(processed_img)
            
            if not self.is_back_side(text):
                if self.debug:
                    print("Back side not detected.")
                return {"address": None, "formatted_output": "Back side not detected"}
            
            address = self.extract_address(text)
            
            if address:
                if self.debug:
                    print("Address extracted successfully.")
                return {"address": address, "formatted_output": f"Address: {address}"}
            
            if self.debug:
                print("No address could be extracted.")
            return {"address": None, "formatted_output": "No address extracted"}
        
        except Exception as e:
            print(f"Error processing image: {e}")
            return {"address": None, "formatted_output": f"Error processing image: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description="Aadhaar Card Address Extractor")
    parser.add_argument("file_path", help="Path to Aadhaar card back side image")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    extractor = AadhaarAddressExtractor(debug=args.debug)
    result = extractor.process(args.file_path)
    
    print("\nExtracted Information:")
    print("="*40)
    print(result["formatted_output"])
    print("="*40)

if __name__ == "__main__":
    main()