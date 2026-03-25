#!/usr/bin/env python3
"""
DINOv3 Canopy Height Model Inference Script
Optimized for multi-GPU inference on large geospatial imagery
"""

# Essential imports for inference only
import os
import warnings
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel
import rasterio
from tiler import Tiler, Merger
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import time
from osgeo import gdal
import traceback
import argparse

# CUDA OPTIMIZATIONS
print("🔧 Setting CUDA optimizations...")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

# Set memory allocation strategy for better multi-GPU performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Environment setup
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

def monitor_gpu_usage():
    """Monitor GPU memory usage during inference"""
    if torch.cuda.is_available():
        print("🔍 GPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            utilization = (allocated / total) * 100
            print(f"  GPU {i}: {allocated:.2f}GB/{total:.1f}GB ({utilization:.1f}%) allocated, {reserved:.2f}GB reserved")
    else:
        print("❌ No CUDA GPUs available for monitoring")

def get_channels(input_bands):
    """Get channel indices based on input band configuration"""
    if input_bands == 'nrg':
        return [3, 2, 1]  # NIR, Red, Green channels
    elif input_bands == 'rgb':
        return [2, 1, 0]  # RGB channels
    else:
        return [0, 1, 2, 3]  # All channels

def create_custom_binned_colormap(colors, vmax=35):
    """Create custom colormap for visualization"""
    boundaries = [0, 0.001, .5, 1, 2, 3, 5, 10, vmax]
    forest_ht_cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, len(colors))
    return forest_ht_cmap, norm, boundaries

class ChannelAttention(nn.Module):
    """Channel attention mechanism to focus on important features"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.attention(x)

class DINOv3DepthHead(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-sat493m", 
             freeze_backbone=True, output_channels=1, token=None, input_bands='rgb',
             training_config=None):
        
        super().__init__()
        
        # Store training config for use in decoder
        self.training_config = training_config or {}
        self.dropout_rate = self.training_config.get('dropout_rate', 0.1)  # Default to 0.1
        
        # Store output_channels for use in decoder
        self.output_channels = output_channels
        
        # Load DINOv3 backbone
        if token:
            self.backbone = AutoModel.from_pretrained(model_name, token=token)
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
    
        # Modify input weights for NRG if specified
        if input_bands == 'nrg':
            self._modify_input_weights_for_nrg()
        
        # Get actual model dimensions dynamically
        self._determine_model_dimensions()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self._freeze_backbone = True
        
        print(f"DINOv3 DepthHead initialized:")
        print(f"  Model: {model_name}")
        print(f"  Input bands: {input_bands}")
        print(f"  Embedding dim: {self.embed_dim}")
        print(f"  Total tokens: {self.total_tokens}")
        print(f"  Spatial patches: {self.num_spatial_patches}")
        print(f"  Using tokens {self.spatial_start}:{self.spatial_end}")
        
        # Build improved decoder
        self.depth_head = self._build_improved_decoder()
    
        # Initialize the final layer
        self._initialize_output_layer(target_percentile=8.42)
    
    def _determine_model_dimensions(self):
        """Dynamically determine model dimensions by running a test forward pass"""
        # Use 64x64 dummy input to match your actual data
        dummy_input = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            outputs = self.backbone(dummy_input)
            features = outputs.last_hidden_state
        
        # Get actual dimensions
        self.total_tokens = features.shape[1]
        self.embed_dim = features.shape[2]
        
        # Calculate spatial patch info for 64x64 input with 16x16 patches
        self.patches_per_side = 64 // 16  # = 4
        self.num_spatial_patches = self.patches_per_side ** 2  # = 16
        
        # Determine token structure
        non_spatial_tokens = self.total_tokens - self.num_spatial_patches
        
        if non_spatial_tokens == 1:
            # Structure: [CLS] + [16 spatial]
            self.spatial_start = 1
            self.spatial_end = self.total_tokens
        elif non_spatial_tokens == 5:
            # Structure: [CLS] + [4 register] + [16 spatial]  
            self.spatial_start = 5
            self.spatial_end = self.total_tokens
        else:
            # Generic: take last 16 tokens as spatial
            self.spatial_start = self.total_tokens - self.num_spatial_patches
            self.spatial_end = self.total_tokens

    def _modify_input_weights_for_nrg(self):
        """Simple weight modification for NIR-Red-Green input"""
        print("Modifying input weights for NIR-Red-Green bands...")
        
        # Find the first Conv2d layer with 3 input channels (this is the patch embedding)
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                print(f"Found patch embedding at: {name}")
                
                with torch.no_grad():
                    original_weights = module.weight.data.clone()
                    
                    # NIR=Red, Red=Red, Green=Green (assuming BGR input order)
                    new_weights = torch.zeros_like(original_weights)
                    new_weights[:, 0, :, :] = original_weights[:, 2, :, :]  # NIR <- Red
                    new_weights[:, 1, :, :] = original_weights[:, 2, :, :]  # Red <- Red  
                    new_weights[:, 2, :, :] = original_weights[:, 1, :, :]  # Green <- Green
                    
                    module.weight.data = new_weights
                    print("✅ Weights modified successfully")
                break

    def _build_improved_decoder(self):
        """Improved decoder adapted for 16x16 input (1x1 spatial features)"""
        
        class ResidualBlock(nn.Module):
            """Residual block with LeakyReLU and attention"""
            def __init__(self, channels, use_attention=True, dropout_rate=0.1):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(channels)
                self.activation = nn.LeakyReLU(0.1, inplace=True)
                
                # Add dropout after activation
                self.dropout = nn.Dropout2d(dropout_rate)
                
                self.attention = ChannelAttention(channels) if use_attention else nn.Identity()
                
            def forward(self, x):
                residual = x
                out = self.activation(self.bn1(self.conv1(x)))
                out = self.dropout(out)  # Apply dropout after activation
                out = self.bn2(self.conv2(out))
                out = self.attention(out)
                out += residual  # Skip connection
                return self.activation(out)
               
        # Get dropout rate from training config
        dropout_rate = self.dropout_rate
        
        layers = nn.ModuleList()
        current_channels = self.embed_dim
        
        # Need 4 upsampling steps: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        target_channels = [512, 256, 128, 64]
        
        for i, out_channels in enumerate(target_channels):
            # Upsampling block (2x upsampling each time)
            upsample_block = nn.Sequential(
                nn.ConvTranspose2d(current_channels, out_channels, 
                                 kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(dropout_rate)
            )
            layers.append(upsample_block)
            
            # Residual refinement blocks with dropout
            refinement_block = nn.Sequential(
                ResidualBlock(out_channels, use_attention=(i >= 2), dropout_rate=dropout_rate),
                ResidualBlock(out_channels, use_attention=(i >= 2), dropout_rate=dropout_rate)
            )
            layers.append(refinement_block)
            
            current_channels = out_channels

        # Final layers
        final_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout_rate),
            ChannelAttention(128),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Final prediction
            nn.Conv2d(32, self.output_channels, kernel_size=1)
        )
        layers.append(final_layers)
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Validate input size
        if x.shape[-2:] != (64, 64):
            raise ValueError(f"Expected input size 64x64, got {x.shape[-2:]}")
        
        # Get features from DINOv3
        if hasattr(self, '_freeze_backbone'):
            with torch.no_grad():
                outputs = self.backbone(x)
        else:
            outputs = self.backbone(x)
            
        features = outputs.last_hidden_state
        
        # Extract spatial tokens
        spatial_tokens = features[:, self.spatial_start:self.spatial_end, :]
        batch_size = spatial_tokens.shape[0]
        
        # Reshape to spatial grid (4x4 for 64x64 input)
        spatial_features = spatial_tokens.transpose(1, 2).reshape(
            batch_size, self.embed_dim, self.patches_per_side, self.patches_per_side
        )
        
        # Pass through decoder (4x4 -> 64x64)
        depth_map = self.depth_head(spatial_features)
        depth_map = depth_map.squeeze(1)
        depth_map = torch.clamp(depth_map, min=0.0)
        
        return depth_map
        
    def _initialize_output_layer(self, target_mean=None, target_percentile=None):
        """Initialize final layer to better predict the full range with skewed data"""
        # Find the final convolutional layer
        final_conv = None
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                final_conv = module
        
        if final_conv is None:
            print("Warning: Could not find final convolutional layer for initialization")
            return
            
        with torch.no_grad():
            nn.init.normal_(final_conv.weight, mean=0.0, std=0.02)
            
            if final_conv.bias is not None:
                if target_mean is not None:
                    final_conv.bias.fill_(target_mean)
                elif target_percentile is not None:
                    final_conv.bias.fill_(target_percentile)
                else:
                    final_conv.bias.fill_(3.0)

def load_model_for_inference(checkpoint_path, device='cuda', use_multi_gpu=True, data_config=None):
    """Load model for inference with multi-GPU support"""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    print(f"Loading {model_config['description']}")
    
    # Create model
    hf_token = "{INSERT YOUR TOKEN HERE}"
    model = DINOv3DepthHead(
        model_name=model_config['model_name'],
        freeze_backbone=True,
        token=hf_token,
        input_bands=data_config['input_bands']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Multi-GPU setup with better configuration
    if use_multi_gpu and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"🚀 Setting up DataParallel across {num_gpus} GPUs!")
        
        # Set up DataParallel
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        
        print(f"✅ DataParallel setup complete:")
        print(f"   Device IDs: {model.device_ids}")
    
    model.eval()
    print(f"✅ Model loaded successfully!")
    return model

def convert_predictions_to_decimeters_int16(predictions):
    """Convert predictions from meters to decimeters, round to nearest integer, 
    and convert to int16 data type. Preserves -9999 NoData values."""
    
    # Create NoData mask
    nodata_mask = predictions == -9999
    
    # Work with valid data only
    valid_predictions = predictions.copy()
    valid_predictions[nodata_mask] = 0  # Temporarily set to 0 for processing
    
    # Multiply by 10 to convert meters to decimeters
    decimeters = valid_predictions * 10
    
    # Round to nearest integer
    rounded = np.round(decimeters)
    
    # Convert to int16
    converted_predictions = rounded.astype(np.int16)

    # Make all negative values = 0
    converted_predictions[converted_predictions < 0] = 0 
    
    # Restore NoData values
    converted_predictions[nodata_mask] = -9999
    
    print(f"Converted to decimeters. NoData pixels preserved: {np.sum(nodata_mask):,}")
    
    return converted_predictions

def save_predictions_as_geotiff(predictions_decimeters, reference_tif_path, output_path):
    """Save predictions as a GeoTIFF with matching geotransform and projection from reference,
    with LZW compression, NoData value of -9999, and metadata about units."""
    
    # Open the reference dataset to get geospatial information
    ref_ds = gdal.Open(reference_tif_path)
    if ref_ds is None:
        raise ValueError(f"Could not open reference file: {reference_tif_path}")
    
    # Get geospatial information from reference
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    
    # Get dimensions
    height, width = predictions_decimeters.shape
    
    # Replace NaN values with -9999 (NoData value)
    output_array = predictions_decimeters.copy()
    if np.issubdtype(predictions_decimeters.dtype, np.floating):
        output_array = np.where(np.isnan(predictions_decimeters), -9999, predictions_decimeters)
    output_array = output_array.astype(np.int16)
    
    # Create the output dataset
    driver = gdal.GetDriverByName('GTiff')
    
    # Create dataset with LZW compression
    out_ds = driver.Create(
        output_path, 
        width, 
        height, 
        1,  # number of bands
        gdal.GDT_Int16,  # data type
        options=['COMPRESS=LZW', 'TILED=YES']  # LZW compression + tiling for efficiency
    )
    
    if out_ds is None:
        raise ValueError(f"Could not create output file: {output_path}")
    
    # Set geospatial information
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    # Get the band and write data
    band = out_ds.GetRasterBand(1)
    band.WriteArray(output_array)
    
    # Set NoData value
    band.SetNoDataValue(-9999)
    
    # Add metadata about units
    band.SetDescription("Canopy Height Model - Values in decimeters")
    band.SetMetadataItem("UNITS", "decimeters")
    band.SetMetadataItem("DESCRIPTION", "Predicted canopy heights in decimeters (1 meter = 10 decimeters)")
    
    # Add dataset-level metadata
    out_ds.SetMetadataItem("PROCESSING", "DINOv3 model prediction")
    out_ds.SetMetadataItem("NODATA_VALUE", "-9999")
    out_ds.SetMetadataItem("UNITS", "decimeters")
    
    # Flush and close
    band.FlushCache()
    out_ds.FlushCache()
    band = None
    out_ds = None
    ref_ds = None
    
    print(f"Successfully saved predictions to: {output_path}")
    print(f"- Compression: LZW")
    print(f"- NoData value: -9999") 
    print(f"- Units: decimeters")
    print(f"- Data type: int16")
    print(f"- Dimensions: {height} x {width}")

def complete_dinov3_inference_pipeline_batch(tif_list, model, data_config, inference_config):
    """Complete pipeline for DINOv3 inference on a list of large geospatial imagery files."""
    
    # Extract configuration values
    means = data_config['means']
    stds = data_config['stds']
    
    nir_min = inference_config['nir_min']
    nir_max = inference_config['nir_max']
    red_min = inference_config['red_min']
    red_max = inference_config['red_max']
    green_min = inference_config['green_min']
    green_max = inference_config['green_max']
    img_size = inference_config['img_size']
    overlap = inference_config['overlap']
    batch_size = inference_config['batch_size']
    visualize_first = inference_config['visualize_first']
    
    print("🚀 Starting Batch DINOv3 Inference Pipeline")
    print("=" * 60)
    
    # Create output directory
    output_dir = inference_config.get('output_dir', 
                                    "/panfs/ccds02/nobackup/projects/above/misc/ABoVE_Shrubs/development/chm/dinov3/4.3.2.6/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created folder: {output_dir}")
    else:
        print(f"Folder already exists: {output_dir}")
    
    print(f"Total TIF files to process: {len(tif_list)}")
    print("=" * 60)
    
    # Set up model
    model.eval()
    device = next(model.parameters()).device
    
    # Create colormap for visualization
    colors = ['#636363','#fc8d59','#fee08b','#ffffbf',
              '#d9ef8b','#91cf60','#1a9850','#005a32']
    forest_ht_cmap, forest_ht_norm, boundaries = create_custom_binned_colormap(colors, vmax=35)
    
    # Initial GPU status
    print("🔄 Initial GPU status:")
    monitor_gpu_usage()
    
    results = []
    output_h, output_w = None, None  # Initialize for model output dimensions
    
    for file_idx, tif_path in enumerate(tif_list, 1):
        print(f"\n🔄 Processing TIF {file_idx}/{len(tif_list)}: {os.path.basename(tif_path)}")
        print("-" * 50)
        
        try:
            # Step 1: Load and preprocess imagery
            print("📂 Step 1: Loading and preprocessing imagery...")
            
            with rasterio.open(tif_path) as src:
                print(f"   Input shape: {src.shape}")
                print(f"   Bands: {src.count}")
                
                # Extract NIR, Red, Green channels (assuming bands 4, 3, 2)
                nir_band = src.read(4)   # NIR
                red_band = src.read(3)   # Red
                green_band = src.read(2) # Green
                
                # Stack into NRG array (channels first)
                nrg_array = np.stack([nir_band, red_band, green_band], axis=0)
            
            print(f"   Extracted NRG shape: {nrg_array.shape}")
            
            # Preprocess data
            nrg_array = nrg_array.astype(np.float32)
            nrg_array[nrg_array == -9999] = np.nan
            
            # Scale to 0-1 range first (like in training)
            nrg_array[0] = (nrg_array[0] - nir_min) / (nir_max - nir_min)
            nrg_array[1] = (nrg_array[1] - red_min) / (red_max - red_min) 
            nrg_array[2] = (nrg_array[2] - green_min) / (green_max - green_min)
            
            print("nir_mean: ", np.nanmean(nrg_array[0]), " red_mean: ", np.nanmean(nrg_array[1]), " green_mean: ", np.nanmean(nrg_array[2]))            
            
            # Clip to ensure 0-1 range
            nrg_array = np.clip(nrg_array, 0, 1)
            
            # Normalize using training statistics
            nrg_array[0] = (nrg_array[0] - means[0]) / stds[0]
            nrg_array[1] = (nrg_array[1] - means[1]) / stds[1]
            nrg_array[2] = (nrg_array[2] - means[2]) / stds[2]
            
            print(f"   ✓ Preprocessing complete")
            print(f"   NaN pixels: {np.isnan(nrg_array).sum():,}")
            
            # Step 2: Set up sliding window inference
            print(f"🧠 Step 2: Running DINOv3 inference...")
            
            # Transpose to (H, W, C) for tiler
            xraster = nrg_array.transpose(1, 2, 0)
            
            # Set up image tiler
            tiler_image = Tiler(
                data_shape=xraster.shape,
                tile_shape=(img_size, img_size, 3),
                channel_dimension=-1,
                overlap=overlap,
                mode='reflect'
            )
            
            # Get model output dimensions (only for first file)
            if file_idx == 1:
                test_input = torch.randn(1, 3, img_size, img_size).to(device)
                with torch.no_grad():
                    test_output = model(test_input)
                    output_h, output_w = test_output.shape[1], test_output.shape[2]
                print(f"   Model output size: {output_h}x{output_w}")
            
            # Set up merger for combining tile predictions
            tiler_mask = Tiler(
                data_shape=(xraster.shape[0], xraster.shape[1], 1),
                tile_shape=(output_h, output_w, 1),
                channel_dimension=-1,
                overlap=overlap,
                mode='reflect'
            )
            merger = Merger(tiler=tiler_mask, window='triang')
            
            # Process tiles in batches
            total_tiles = len(tiler_image)
            print(f"   Processing {total_tiles:,} tiles with batch size {batch_size}")
            start_time = time.time()
            tile_count = 0
            
            for batch_id, batch_i in tiler_image(xraster, batch_size=batch_size):
                actual_batch_size = len(batch_i)
                
                # Prepare input batch
                input_batch = batch_i.transpose(0, 3, 1, 2).astype('float32')
                input_batch_tensor = torch.from_numpy(input_batch).to(device)
                input_batch_tensor = torch.nan_to_num(input_batch_tensor, nan=0.0)
                
                # Run inference
                with torch.no_grad():
                    y_batch = model(input_batch_tensor)
                    y_batch_numpy = y_batch.cpu().numpy()
                    
                    # Clear GPU memory immediately
                    del input_batch_tensor, y_batch
                    
                    # Format for merger
                    if len(y_batch_numpy.shape) == 3:
                        formatted_output = np.expand_dims(y_batch_numpy, axis=-1)
                    else:
                        formatted_output = y_batch_numpy
                        
                # Add predictions to merger
                for j in range(actual_batch_size):
                    tile_id = batch_id * batch_size + j
                    merger.add(tile_id, formatted_output[j])
                
                tile_count += actual_batch_size
                
                # Progress update
                if tile_count % (batch_size * 50) == 0:
                    progress = (tile_count / total_tiles) * 50
                    elapsed_time = time.time() - start_time
                    tiles_per_sec = tile_count / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (total_tiles - tile_count) / tiles_per_sec if tiles_per_sec > 0 else 0
                    
                    print(f"     Progress: {progress:.1f}% ({tile_count:,}/{total_tiles:,}) | "
                          f"Speed: {tiles_per_sec:.0f} tiles/sec | ETA: {eta_seconds/60:.1f} min")
                
                # GPU monitoring
                if tile_count % (batch_size * 250) == 0:
                    monitor_gpu_usage()
                
                # Memory management
                if tile_count % (batch_size * 50) == 0:
                    torch.cuda.empty_cache()
            
            print(f"✅ Completed processing {tile_count:,} tiles total")
            
            # Merge tile predictions
            print("   🔄 Merging tiles...")
            predictions_raw = merger.merge(unpad=True)
            predictions_meters = np.squeeze(predictions_raw)
            
            print(f"   ✓ Inference complete")
            print(f"   Final prediction shape: {predictions_meters.shape}")
            
            # GPU status after inference
            print("🔄 GPU status after inference:")
            monitor_gpu_usage()
            
            # Step 3: Post-process predictions
            print(f"🔧 Step 3: Post-processing predictions...")
            
            # Restore original NoData locations
            original_nodata = np.isnan(nrg_array[0])
            predictions_meters[original_nodata] = -9999
            
            # Convert to decimeters (int16 for storage efficiency)
            predictions_decimeters = convert_predictions_to_decimeters_int16(predictions_meters)
            print(f"   ✓ Converted to decimeters (int16)")
            
            # Step 4: Save as GeoTIFF
            print(f"💾 Step 4: Saving GeoTIFF...")
            
            # Generate output filename
            base_name = os.path.basename(tif_path)
            if len(base_name) >= 46:
                string = base_name[-46:-10]
            else:
                string = base_name[:-4]
            output_tif = os.path.join(output_dir, f'{string}sr-02m.chm.tif')
            
            save_predictions_as_geotiff(predictions_decimeters, tif_path, output_tif)
            
            # Step 5: Optional visualization (disabled for headless server)
            if visualize_first and file_idx <= 3 and os.environ.get('DISPLAY'):
                print(f"📊 Step 5: Generating visualization for file {file_idx}...")
                
                plt.figure(figsize=(20, 6))
                
                # Original NRG composite
                plt.subplot(1, 3, 1)
                nrg_display = nrg_array.transpose(1, 2, 0)
                display_step = max(1, min(nrg_display.shape[0], nrg_display.shape[1]) // 1000)
                nrg_sub = nrg_display[::display_step, ::display_step, :]
                
                # Normalize for display
                nrg_norm = np.zeros_like(nrg_sub)
                for ch in range(3):
                    channel = nrg_sub[:, :, ch]
                    valid = ~np.isnan(channel)
                    if valid.any():
                        vmin, vmax = np.nanpercentile(channel[valid], [2, 98])
                        if vmax > vmin:
                            nrg_norm[:, :, ch] = np.clip((channel - vmin)/(vmax - vmin), 0, 1)
                
                plt.imshow(nrg_norm)
                plt.title(f'NRG Composite\n{os.path.basename(tif_path)}')
                plt.axis('off')
                
                # Predicted CHM
                plt.subplot(1, 3, 2)
                pred_display = predictions_meters.copy()
                pred_display[pred_display == -9999] = np.nan
                pred_sub = pred_display[::display_step, ::display_step]
                
                valid_pred = pred_display[~np.isnan(pred_display)]
                if len(valid_pred) > 0:
                    vmin, vmax = np.nanpercentile(valid_pred, [1, 99])
                    im1 = plt.imshow(pred_sub, cmap=forest_ht_cmap, vmin=vmin, vmax=vmax)
                    plt.colorbar(im1, label='Height (m)')
                else:
                    plt.imshow(pred_sub, cmap='viridis')
                
                plt.title('Predicted CHM (meters)')
                plt.axis('off')
                
                # Statistics histogram
                plt.subplot(1, 3, 3)
                if len(valid_pred) > 0:
                    plt.hist(valid_pred, bins=50, alpha=0.7, edgecolor='black')
                    plt.xlabel('Predicted Height (m)')
                    plt.ylabel('Frequency')
                    plt.title(f'Height Distribution\nMean: {valid_pred.mean():.2f}m')
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/visualization_file_{file_idx:03d}.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()  # Close figure to save memory
            
            # Add to results
            results.append(predictions_meters)
            
            # Print summary for this file
            valid_pixels = (predictions_meters != -9999).sum()
            nodata_pixels = (predictions_meters == -9999).sum()
            
            print(f"✅ Completed TIF {file_idx}/{len(tif_list)}")
            print(f"   📊 Valid predictions: {valid_pixels:,} pixels")
            print(f"   🚫 NoData pixels: {nodata_pixels:,} pixels")
            
            if valid_pixels > 0:
                valid_heights = predictions_meters[predictions_meters != -9999]
                print(f"   🌲 Height range: {valid_heights.min():.2f}m to {valid_heights.max():.2f}m")
                print(f"   📈 Mean height: {valid_heights.mean():.2f}m ± {valid_heights.std():.2f}m")
            
        except Exception as e:
            print(f"❌ Error processing {tif_path}: {str(e)}")
            traceback.print_exc()
            results.append(None)
            continue
    
    # Print final summary
    successful_files = sum(1 for r in results if r is not None)
    failed_files = len(results) - successful_files
    
    print("\n" + "="*60)
    print("⭐ BATCH INFERENCE COMPLETE ⭐")
    print("="*60)
    print(f"📁 Output directory: {output_dir}")
    print(f"✅ Successfully processed: {successful_files}/{len(tif_list)} files")
    if failed_files > 0:
        print(f"❌ Failed: {failed_files} files")
    
    # Final GPU monitoring
    print("🔄 Final GPU status after all processing:")
    monitor_gpu_usage()
    print("="*60)
    
    return results

def load_file_list(file_list_path=None):
    """Load list of files to process"""
    if file_list_path and os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        print(f"📋 Loaded {len(files)} files from {file_list_path}")
        return files
    else:
        # Fallback to hardcoded list
        print("⚠️  File list not found, using fallback list")
        return [
            '/panfs/ccds02/nobackup/projects/above/misc/ABoVE_Shrubs/srlite/002m_old/WV02_20190707_M1BS_103001009433EB00-sr-02m.tif',
            '/panfs/ccds02/nobackup/projects/above/misc/ABoVE_Shrubs/srlite/002m_old/WV02_20190707_M1BS_103001009444A100-sr-02m.tif',
            '/panfs/ccds02/nobackup/projects/above/misc/ABoVE_Shrubs/srlite/002m_old/WV02_20190707_M1BS_10300100944FA100-sr-02m.tif'
        ]

def check_output_exists(tif_path, output_dir):
    """Check if output file already exists for a given input TIF"""
    base_name = os.path.basename(tif_path)
    
    # Simple replacement: .tif -> .chm.tif
    if base_name.endswith('.tif'):
        output_filename = base_name[:-4] + '.chm.tif'
    else:
        output_filename = base_name + '.chm'
    
    output_path = os.path.join(output_dir, output_filename)
    return os.path.exists(output_path), output_path

def filter_unprocessed_files(tif_list, output_dir):
    """Filter out files that have already been processed"""
    unprocessed_files = []
    skipped_count = 0
    
    print(f"🔍 Checking which files need processing...")
    
    for tif_path in tif_list:
        exists, expected_output = check_output_exists(tif_path, output_dir)
        if exists:
            print(f"  ⏭️  Already exists: {os.path.basename(expected_output)}")
            skipped_count += 1
        else:
            unprocessed_files.append(tif_path)
    
    print(f"📊 Found {len(unprocessed_files)} files to process, {skipped_count} already completed")
    return unprocessed_files

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DINOv3 CHM Inference Script')
    
    parser.add_argument('--file-list', type=str, 
                       default='/explore/nobackup/people/mfrost2/temp/files_to_inference.txt',
                       help='Path to file containing list of TIFs to process')
    
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for inference')
    
    parser.add_argument('--output-dir', type=str, 
                       default='/panfs/ccds02/nobackup/projects/above/misc/ABoVE_Shrubs/development/chm/dinov3/4.3.2.6/',
                       help='Output directory for results')
    
    parser.add_argument('--checkpoint', type=str,
                       default='best_dinov3_fulldataset_customloss.pth',
                       help='Path to model checkpoint')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots (requires display)')
    
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Tile overlap for inference (0.0-1.0)')
    
    parser.add_argument('--img-size', type=int, default=64,
                       help='Input image size for model')
    
    return parser.parse_args()

def setup_data_config():
    """Setup data configuration"""
    DATA_CONFIG = {
        'data_name': 'chm_npy_dataset',
        'stats_path': '/explore/nobackup/people/mfrost2/projects/boreal_chm_dino/numpy_stats/',  
        'np_stats': 'maxmin_ak_100k_both_nrg_final',
        'nir_min': 0, 'nir_max': 7142,
        'red_min': 0, 'red_max': 5893,
        'green_min': 0, 'green_max': 5387,
        'input_bands': 'nrg'
    }
    
    # Load normalization stats
    DATA_CONFIG['means'] = np.load(f"{DATA_CONFIG['stats_path']}channel_means_{DATA_CONFIG['np_stats']}.npy")
    DATA_CONFIG['stds'] = np.load(f"{DATA_CONFIG['stats_path']}channel_stds_{DATA_CONFIG['np_stats']}.npy")
    
    print(f"Input bands: {DATA_CONFIG['input_bands']}")
    print(f"Channel means: {DATA_CONFIG['means']}")
    print(f"Channel stds: {DATA_CONFIG['stds']}")
    
    return DATA_CONFIG

def setup_inference_config(args):
    """Setup inference configuration from arguments"""
    INFERENCE_CONFIG = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'overlap': args.overlap,
        'visualize_first': args.visualize,
        'output_dir': args.output_dir,
        'blue_min': 0, 'blue_max': 5681,
        'nir_min': 0, 'nir_max': 7142,
        'red_min': 0, 'red_max': 5893,
        'green_min': 0, 'green_max': 5387,
    }
    
    print(f"🚀 Using device: {INFERENCE_CONFIG['device']}")
    print(f"🚀 Using batch size: {INFERENCE_CONFIG['batch_size']}")
    print(f"🚀 Image size: {INFERENCE_CONFIG['img_size']}")
    print(f"🚀 Overlap: {INFERENCE_CONFIG['overlap']}")
    print(f"🚀 Output directory: {INFERENCE_CONFIG['output_dir']}")
    
    return INFERENCE_CONFIG

def main():
    """Main execution function"""
    print("🌲 DINOv3 Canopy Height Model Inference")
    print("=" * 60)
    
    # Parse command line arguments
    args = parse_args()
    
    # Setup configurations
    DATA_CONFIG = setup_data_config()
    INFERENCE_CONFIG = setup_inference_config(args)
    
    # Load file list
    tif_list = load_file_list(args.file_list)
    
    if not tif_list:
        print("❌ No files to process!")
        return
    
    # Add this section to filter files
    # ----------------------------------------------------------------
    # Filter out already processed files
    unprocessed_files = filter_unprocessed_files(tif_list, INFERENCE_CONFIG['output_dir'])
    
    if not unprocessed_files:
        print("✅ All files have already been processed!")
        return
    # ----------------------------------------------------------------
    
    # Load model
    print(f"\n📦 Loading model from: {args.checkpoint}")
    model = load_model_for_inference(
        args.checkpoint, 
        INFERENCE_CONFIG['device'], 
        use_multi_gpu=True,
        data_config=DATA_CONFIG
    )
    
    # Change this line to use unprocessed_files instead of tif_list
    # ----------------------------------------------------------------
    # Run batch inference on unprocessed files only
    if model is not None:
        batch_start_time = time.time()
        print(f"\n🚀 Starting batch inference on {len(unprocessed_files)} unprocessed files...")
        
        batch_results = complete_dinov3_inference_pipeline_batch(
            tif_list=unprocessed_files,  # Changed this line
            model=model,
            data_config=DATA_CONFIG,
            inference_config=INFERENCE_CONFIG
        )
        
        batch_end_time = time.time()
        total_batch_time = batch_end_time - batch_start_time
        
        print(f"\n🎉 BATCH PROCESSING COMPLETE!")
        print(f"⏰ Total time: {total_batch_time:.1f} seconds ({total_batch_time/60:.1f} minutes)")
        print(f"📊 Average per file: {total_batch_time/len(unprocessed_files):.1f} seconds")
        
    else:
        print("❌ No model available for inference")

if __name__ == "__main__":
    main()