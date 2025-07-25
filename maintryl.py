````````try:
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