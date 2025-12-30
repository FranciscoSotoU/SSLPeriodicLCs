# Flexible Positional Encoder

A highly configurable positional encoder for time series data that allows switching between different embedding strategies based on model parameters and requirements.

## Features

The `FlexiblePositionalEncoder` supports multiple encoding strategies and can be configured to use different combinations of embeddings:

### Encoding Strategies

1. **Basic**: Simple absolute embeddings (sinusoidal time + convolutional magnitude)
2. **Enhanced**: Includes time/magnitude differences and rates of change
3. **Advanced**: Full feature set with multiple representations
4. **Custom**: User-defined combination of features

### Available Embeddings

- **Sinusoidal Time Encoding**: Traditional sinusoidal positional encoding for time
- **Convolutional Magnitude**: 1D convolution for magnitude values
- **Time Differences**: MLP-based encoding of time differences between consecutive points
- **Magnitude Differences**: MLP-based encoding of magnitude differences
- **Rate of Change**: Encoding of magnitude_diff / time_diff
- **Absolute Time MLP**: Alternative MLP-based time encoding
- **Absolute Magnitude MLP**: Alternative MLP-based magnitude encoding
- **Band Embedding**: Learnable embeddings for photometric bands

### Fusion Strategies

- **MLP**: Multi-layer perceptron for combining embeddings (default, configurable depth)
- **Attention**: Self-attention mechanism for fusion
- **Simple**: Linear projection with normalization

### MLP Configuration

The MLP fusion strategy now supports configurable depth:

- **1 layer**: Direct mapping from concatenated features to output
- **2-3 layers**: Good balance for most use cases (default: 2)
- **4+ layers**: Deep fusion for complex feature interactions

## Usage

### Quick Start

```python
from flexible_positional_encoder import (
    FlexiblePositionalEncoder,
    create_basic_encoder,
    create_enhanced_encoder,
    create_advanced_encoder
)

# Basic encoder (minimal features)
encoder = create_basic_encoder(embedding_size=128)

# Enhanced encoder (recommended for most use cases)
encoder = create_enhanced_encoder(
    embedding_size=256,
    num_bands=2,
    use_differences=True,
    use_rate=True,
    mlp_layers=3  # Deeper fusion
)

# Advanced encoder (all features)
encoder = create_advanced_encoder(
    embedding_size=512,
    fusion_strategy='attention',
    mlp_layers=4  # Deep MLP when using attention
)
```

### Custom Configuration

```python
# Define custom features
custom_features = {
    'sinusoidal_time': True,
    'conv_magnitude': True,
    'time_differences': True,
    'mag_differences': False,  # Disable magnitude differences
    'rate_of_change': False,   # Disable rate calculation
    'abs_time_mlp': True,
    'abs_mag_mlp': True,
    'band_embedding': True
}

encoder = FlexiblePositionalEncoder(
    embedding_size=256,
    encoding_strategy='custom',
    features_config=custom_features,
    fusion_strategy='mlp',
    mlp_layers=3,  # Custom depth
    reduced_size_factor=2
)
```

### Using Preset Configurations

```python
from encoder_config import get_preset_config, create_encoder_from_config

# Get a preset configuration
config = get_preset_config('lightcurve_medium')

# Create encoder from config
encoder_kwargs = create_encoder_from_config(config)
encoder = FlexiblePositionalEncoder(**encoder_kwargs)
```

### Forward Pass

```python
import torch

# Prepare input data
batch_size, seq_len = 4, 100
x = torch.randn(batch_size, seq_len, 1)  # magnitude values
t = torch.randn(batch_size, seq_len, 1)  # time values
bands = torch.randint(0, 2, (batch_size, seq_len))  # band information

# Forward pass
output = encoder(x, t, bands)  # Shape: [batch_size, seq_len, embedding_size]
```

## Configuration Options

### Memory Efficiency

Control memory usage with the `reduced_size_factor` parameter:

```python
# More memory efficient (smaller intermediate embeddings)
encoder = create_enhanced_encoder(
    embedding_size=512,
    reduced_size_factor=4  # Uses 512/4 = 128 for intermediate embeddings
)
```

### Dropout and Regularization

```python
encoder = FlexiblePositionalEncoder(
    embedding_size=256,
    dropout=0.2,  # Higher dropout for regularization
    mlp_layers=2,  # Moderate depth
    encoding_strategy='enhanced'
)
```

## Integration with Existing Models

### With ATAT Module

```python
# In your model configuration
from encoder_config import integrate_with_atat_model

model_config = {
    'use_lightcurve': True,
    'lc_embedding_size': 256,
    'model_type': 'transformer',
    'memory_efficient': True
}

# Automatically configure encoder based on model parameters
model_config = integrate_with_atat_model(model_config)
encoder_kwargs = model_config['positional_encoder']

# Create encoder
encoder = FlexiblePositionalEncoder(**encoder_kwargs)
```

### Replacing PositionalEncoderEnhanced

To replace the existing `PositionalEncoderEnhanced` with equivalent functionality:

```python
# Old way
from positional_encoder_enhanced import PositionalEncoderEnhanced
old_encoder = PositionalEncoderEnhanced(embedding_size=256)

# New way (equivalent)
from flexible_positional_encoder import create_advanced_encoder
new_encoder = create_advanced_encoder(embedding_size=256)
```

## Available Presets

- `lightcurve_small`: Basic features, 128 dimensions
- `lightcurve_medium`: Enhanced features, 256 dimensions  
- `lightcurve_large`: Advanced features, 512 dimensions
- `transformer_*`: Optimized for transformer models
- `convnet_*`: Optimized for convolutional networks
- `ssl_*`: Optimized for self-supervised learning

## Performance Considerations

### Model Size Comparison

| Strategy | Embedding Size | Reduced Factor | MLP Layers | Approx. Parameters | Memory (MB) |
|----------|---------------|----------------|------------|-------------------|-------------|
| Basic    | 128           | 2             | 1          | ~50K              | 0.2         |
| Enhanced | 256           | 2             | 2          | ~200K             | 0.8         |
| Enhanced | 256           | 2             | 4          | ~250K             | 1.0         |
| Advanced | 512           | 4             | 3          | ~500K             | 2.0         |

### Recommendations

- **For small models (<100M params)**: Use `basic` or `enhanced` strategy with 1-2 MLP layers
- **For large models (>100M params)**: Use `advanced` strategy with 3-4 MLP layers
- **For memory-constrained environments**: Set `reduced_size_factor=8`, `mlp_layers=1`, and `fusion_strategy='simple'`
- **For maximum performance**: Use `fusion_strategy='attention'` or deep MLP with 4+ layers

## Examples

See `flexible_encoder_examples.py` for comprehensive usage examples including:

- Basic, enhanced, and advanced configurations
- Custom feature combinations
- Parameter studies
- Memory efficiency comparisons

## Migration Guide

### From PositionalEncoderEnhanced

1. Replace import:
   ```python
   # Old
   from positional_encoder_enhanced import PositionalEncoderEnhanced
   
   # New
   from flexible_positional_encoder import create_advanced_encoder
   ```

2. Update instantiation:
   ```python
   # Old
   encoder = PositionalEncoderEnhanced(embedding_size=256, dropout=0.1)
   
   # New
   encoder = create_advanced_encoder(embedding_size=256, dropout=0.1)
   ```

3. Forward pass remains the same:
   ```python
   output = encoder(x, t, bands)
   ```

### Configuration Benefits

The new flexible encoder provides:

- **Better memory control** with `reduced_size_factor`
- **Feature selection** to disable unused embeddings
- **Multiple fusion strategies** for different model types
- **Preset configurations** for common use cases
- **Easy integration** with existing codebases

## Debugging

Use `encoder.get_config()` to inspect the current configuration:

```python
config = encoder.get_config()
print(f"Features enabled: {config['features']}")
print(f"Strategy: {config['encoding_strategy']}")
print(f"Fusion: {config['fusion_strategy']}")
```
