import json
import os
import sys

# Add the model_fetcher directory to sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "model_fetcher"))

import llm_hw

def test_snapshot_deduplication():
    print("Testing llm_hw.snapshot() output...")
    snap = llm_hw.snapshot()
    
    print(f"OS: {snap.get('os')}")
    print(f"CPU: {snap.get('cpu')}")
    print(f"RAM: {snap.get('ram_bytes')}")
    
    nv_gpus = snap.get("nvidia_gpus", [])
    amd_gpus = snap.get("amd_gpus", [])
    display_adapters = snap.get("display_adapters", [])
    
    print("\nNVIDIA GPUs detected:")
    for g in nv_gpus:
        print(f"  - {g['name']}")
        
    print("\nAMD GPUs detected:")
    for g in amd_gpus:
        print(f"  - {g['name']}")
        
    print("\nDisplay Adapters (Windows only):")
    for g in display_adapters:
        print(f"  - {g['name']}")

    # Check for duplicates
    all_names = [g['name'] for g in nv_gpus] + [g['name'] for g in amd_gpus]
    duplicates = []
    for da in display_adapters:
        if da['name'] in all_names:
            duplicates.append(da['name'])
            
    if duplicates:
        print(f"\nFound duplicates in display_adapters: {duplicates}")
    else:
        print("\nNo direct name duplicates found between specific GPU lists and display_adapters.")

if __name__ == "__main__":
    test_snapshot_deduplication()
