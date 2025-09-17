#!/usr/bin/env python3
"""
Debug script to identify the exact error location in CPRAformer
"""

import torch
import torch.nn as nn
import traceback
from ultralytics.nn.modules.cpraformer import CPRAformerC2f, LightCPRAformerC2f

def debug_cpraformer():
    print("🔍 Debugging CPRAformer modules...")
    
    # Test with small dimensions first
    x = torch.randn(1, 128, 16, 16)  # Smaller input
    print(f"Input shape: {x.shape}")
    
    try:
        # Test lightweight version first
        print("\n--- Testing LightCPRAformerC2f ---")
        light_cpra = LightCPRAformerC2f(128, 128, n=1)
        print("LightCPRAformerC2f created successfully")
        
        out_light = light_cpra(x)
        print(f"LightCPRAformerC2f: Input {x.shape} -> Output {out_light.shape}")
        
    except Exception as e:
        print(f"❌ LightCPRAformerC2f failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
    
    try:
        # Test full version 
        print("\n--- Testing CPRAformerC2f ---")
        cpra_full = CPRAformerC2f(128, 128, n=1)
        print("CPRAformerC2f created successfully")
        
        out_full = cpra_full(x)
        print(f"CPRAformerC2f: Input {x.shape} -> Output {out_full.shape}")
        
    except Exception as e:
        print(f"❌ CPRAformerC2f failed: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_cpraformer()