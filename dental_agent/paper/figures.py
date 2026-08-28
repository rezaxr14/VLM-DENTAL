import os
from pathlib import Path
from PIL import Image, ImageDraw

def create_figures():
    os.makedirs("data/traces/analysis_charts/case_study", exist_ok=True)
    
    # Try to load image 462 (or any dentex image if 462 is missing)
    # The image path from the trace was likely data/dentex/training_data/quadrant/images/462.png or similar.
    # Let's search for it.
    img_path = None
    for p in Path("data").rglob("*462*.png"):
        if p.is_file():
            img_path = p
            break
            
    if not img_path:
        # Fallback to creating a dummy image for the plan if we can't find 462
        print("Image 462 not found. Generating dummy representations.")
        img = Image.new("RGB", (3000, 1500), color=(50, 50, 50))
    else:
        img = Image.open(img_path).convert("RGB")
        
    # Turn 2: locate_tooth returns bbox [676.0, 475.0, 160.0, 259.0]
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    bbox1 = [676.0, 475.0, 676.0+160.0, 475.0+259.0]
    draw.rectangle(bbox1, outline="red", width=10)
    img1.save("data/traces/analysis_charts/case_study/turn2_locate.png")
    
    # Turn 3: zoom_crop with padding_frac 0.2
    # The agent does zoom_crop(bbox, padding_frac=0.2)
    # Let's just crop it manually for the figure
    x, y, w, h = 676.0, 475.0, 160.0, 259.0
    px, py = w * 0.2, h * 0.2
    crop_box = [max(0, x-px), max(0, y-py), min(img.width, x+w+px), min(img.height, y+h+py)]
    img2 = img.crop(crop_box)
    img2.save("data/traces/analysis_charts/case_study/turn3_zoom.png")
    
    # Turn 4/5: nudge_crop and zoom_crop
    # new bbox: [662.0, 521.2, 184.0, 297.9], padding 0.1
    x, y, w, h = 662.0, 521.2, 184.0, 297.9
    px, py = w * 0.1, h * 0.1
    crop_box2 = [max(0, x-px), max(0, y-py), min(img.width, x+w+px), min(img.height, y+h+py)]
    img3 = img.crop(crop_box2)
    img3.save("data/traces/analysis_charts/case_study/turn5_zoom_nudged.png")
    
    # Turn 6: Contralateral Compare
    # The agent compares the confirmed impaction with the opposite side
    # We mirror the x-coordinate across the center of the image
    center_x = img.width / 2
    dist_to_center = center_x - (x + w/2)
    mirrored_center_x = center_x + dist_to_center
    mirrored_x = mirrored_center_x - w/2
    
    crop_box_mirror = [max(0, mirrored_x-px), max(0, y-py), min(img.width, mirrored_x+w+px), min(img.height, y+h+py)]
    img_mirror = img.crop(crop_box_mirror)
    
    # Create side-by-side composite
    composite = Image.new("RGB", (img3.width + img_mirror.width, max(img3.height, img_mirror.height)))
    composite.paste(img3, (0, 0))
    composite.paste(img_mirror, (img3.width, 0))
    composite.save("data/traces/analysis_charts/case_study/turn6_contralateral.png")
    
    # ---------------------------------------------
    # Case Study 2: Image 401 Empty Crop Recovery
    # ---------------------------------------------
    img_401_path = None
    for p in Path("data").rglob("*401*.png"):
        if p.is_file():
            img_401_path = p
            break
            
    if not img_401_path:
        print("Image 401 not found.")
        img_401 = Image.new("RGB", (3000, 1500), color=(50, 50, 50))
    else:
        img_401 = Image.open(img_401_path).convert("RGB")
        
    # Turn 9: Initial Zoom (Empty Crop)
    x, y, w, h = 1064.4, 717.4, 133.0, 308.0
    px, py = w * 0.15, h * 0.15
    crop_box_401 = [max(0, x-px), max(0, y-py), min(img_401.width, x+w+px), min(img_401.height, y+h+py)]
    img_401_zoom = img_401.crop(crop_box_401)
    img_401_zoom.save("data/traces/analysis_charts/case_study/trace_1_turn9_zoom_empty.png")
    
    # Turn 13: Nudged Zoom
    nx, ny, nw, nh = 1044.5, 794.4, 93.1, 215.6
    npx, npy = nw * 0.15, nh * 0.15
    crop_box_401_nudge = [max(0, nx-npx), max(0, ny-npy), min(img_401.width, nx+nw+npx), min(img_401.height, ny+nh+npy)]
    img_401_nudged = img_401.crop(crop_box_401_nudge)
    img_401_nudged.save("data/traces/analysis_charts/case_study/trace_1_turn13_zoom_nudged.png")
    
    print("Successfully generated case study images.")

if __name__ == "__main__":
    create_figures()
