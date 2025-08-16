import exiftool
import json

def get_exif_data(image_path):
    """
    Extracts and prints EXIF metadata from the given image file.
    """
    with exiftool.ExifTool() as et:
        metadata = et.execute_json("-j", image_path)  # Use execute_json() to get structured data

    if metadata:
        print(json.dumps(metadata[0], indent=4))  # Pretty print JSON output
    else:
        print("No EXIF data found.")

if __name__ == "__main__":
    image_path = input("Enter the image path: ").strip()
    get_exif_data(image_path)
