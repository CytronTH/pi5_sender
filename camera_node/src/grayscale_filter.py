import cv2
import argparse
import os

def convert_to_grayscale(input_dir, output_dir, recursive=False, use_clahe=False):
    """
    Batch convert images in input_dir to grayscale and save to output_dir.
    Optionally apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing images from '{input_dir}'...")
    print(f"Saving grayscale images to '{output_dir}'...")
    if use_clahe:
        print("Using CLAHE (Contrast Limited Adaptive Histogram Equalization)...")

    # Define CLAHE explicitly (clipLimit=2.0 is standard for inspection)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    count = 0
    
    if recursive:
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(image_extensions):
                    input_path = os.path.join(root, f)
                    
                    # Calculate relative path to maintain structure
                    rel_path = os.path.relpath(root, input_dir)
                    target_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    output_path = os.path.join(target_dir, f)
                    
                    try:
                        img = cv2.imread(input_path)
                        if img is None: continue
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        if use_clahe:
                            gray = clahe_obj.apply(gray)
                            
                        cv2.imwrite(output_path, gray)
                        count += 1
                        if count % 100 == 0: print(f"  Processed {count} images...")
                    except Exception as e:
                        print(f"Error: {e}")
    else:
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
        for f in files:
            input_path = os.path.join(input_dir, f)
            output_path = os.path.join(output_dir, f)
            try:
                img = cv2.imread(input_path)
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                if use_clahe:
                    gray = clahe_obj.apply(gray)
                    
                cv2.imwrite(output_path, gray)
                count += 1
                if count % 10 == 0: print(f"  Processed {count}...")
            except Exception as e:
                print(f"Error: {e}")

    print(f"Done! Successfully converted {count} images to grayscale (CLAHE={use_clahe}).")

def main():
    parser = argparse.ArgumentParser(description="Batch convert images to Grayscale.")
    parser.add_argument("input_dir", help="Path to input folder containing images")
    parser.add_argument("--output_dir", help="Path to output folder (default: input_dir_gray)", default=None)
    parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories recursively")
    parser.add_argument("--clahe", action="store_true", help="Apply Contrast Limited Adaptive Histogram Equalization")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        suffix = "_clahe" if args.clahe else "_gray"
        args.output_dir = args.input_dir.rstrip("/\\") + suffix
        
    convert_to_grayscale(args.input_dir, args.output_dir, args.recursive, args.clahe)

if __name__ == "__main__":
    main()
