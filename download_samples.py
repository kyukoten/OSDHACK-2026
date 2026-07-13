import os
import urllib.request

# Define diverse test images from public, un-blocked URLs
test_images = {
    "dog_in_park.jpg": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?q=80&w=600&auto=format&fit=crop",
    "beach_sunset.jpg": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=600&auto=format&fit=crop",
    "red_sports_car.jpg": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=600&auto=format&fit=crop",
    "cat_sleeping.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=600&auto=format&fit=crop",
    "coffee_cup_on_desk.jpg": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?q=80&w=600&auto=format&fit=crop"
}

target_dir = "./sample_images"
os.makedirs(target_dir, exist_ok=True)

print("Downloading test dataset...")
for filename, url in test_images.items():
    filepath = os.path.join(target_dir, filename)
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

print(f"\n✅ Ready! 5 images saved inside '{target_dir}' folder.")