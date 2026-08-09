import os
import shutil
from pathlib import Path
from dental_agent.data.dentex import download_dentex, extract_dentex_zips

def main():
    cache_dir = r"C:\Users\rezax\dental_agent_cache"
    target_data_dir = Path("data/dentex")
    
    print(f"Resuming download using cache directory: {cache_dir}")
    
    # 1. Resume download using the existing cache folder
    # This will pick up exactly where the .incomplete file left off
    downloaded_path = download_dentex(
        repo_id="ibrahimhamamci/DENTEX",
        cache_dir=cache_dir,
        split_name="train"
    )
    
    print("\nDownload complete! Extracting zip files...")
    
    # 2. Extract zips
    extract_dentex_zips(downloaded_path)
    
    # 3. Move the extracted files into our local data directory
    print(f"\nMoving extracted dataset to {target_data_dir}...")
    target_data_dir.mkdir(parents=True, exist_ok=True)
    
    # The downloaded path will contain the extracted 'DENTEX' folder
    # We want to merge its contents into data/dentex/DENTEX/
    source_dentex = downloaded_path / "DENTEX"
    dest_dentex = target_data_dir / "DENTEX"
    
    if source_dentex.exists():
        # Copy tree, merging directories if they already exist
        shutil.copytree(source_dentex, dest_dentex, dirs_exist_ok=True)
        print(f"Successfully moved files to {dest_dentex}")
    else:
        print(f"Warning: Expected DENTEX folder at {source_dentex} but didn't find it.")
        
    # 4. Clean up the massive cache folder
    print(f"\nCleaning up cache directory {cache_dir} to free up 10GB of space...")
    try:
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("Cache cleanup successful!")
    except Exception as e:
        print(f"Could not completely remove cache dir automatically. You can manually delete {cache_dir} safely. Error: {e}")
        
    print("\nAll done! The full dataset is now in data/dentex and the cache is cleared.")

if __name__ == "__main__":
    main()
