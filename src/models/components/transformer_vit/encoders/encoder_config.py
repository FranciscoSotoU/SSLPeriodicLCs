"""
Configuration helper for FlexiblePositionalEncoder integration

This module provides configuration utilities and presets for integrating
the FlexiblePositionalEncoder with different model architectures and use cases.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    """Configuration class for FlexiblePositionalEncoder."""
    embedding_size: int
    encoding_strategy: str = 'enhanced'
    reduced_size_factor: int = 2
    dropout: float = 0.1
    fusion_strategy: str = 'mlp'
    mlp_layers: int = 2
    use_sinusoidal: bool = True
    use_conv_mag: bool = True
    use_differences: bool = True
    use_rate: bool = True
    use_band_embedding: bool = True
    use_abs_time_mlp: Optional[bool] = None
    use_abs_mag_mlp: Optional[bool] = None
    num_bands: int = 2
    seq_length: int = 2048
    features_config: Optional[Dict[str, bool]] = None


class EncoderConfigManager:
    """Manager class for encoder configurations."""
    
    @staticmethod
    def get_lightcurve_config(model_size: str = 'medium') -> EncoderConfig:
        """Get configuration optimized for lightcurve data."""
        configs = {
            'small': EncoderConfig(
                embedding_size=128,
                encoding_strategy='basic',
                reduced_size_factor=2,
                dropout=0.1,
                fusion_strategy='simple',
                mlp_layers=1,  # Simple single layer
                use_differences=False,
                use_rate=False,
                use_abs_time_mlp=False,  # Explicitly disable for basic
                use_abs_mag_mlp=False
            ),
            'medium': EncoderConfig(
                embedding_size=256,
                encoding_strategy='enhanced',
                reduced_size_factor=2,
                dropout=0.15,
                fusion_strategy='mlp',
                mlp_layers=2,  # Standard depth
                use_differences=True,
                use_rate=True,
                use_abs_time_mlp=None,  # Auto-configure based on strategy
                use_abs_mag_mlp=None
            ),
            'large': EncoderConfig(
                embedding_size=512,
                encoding_strategy='advanced',
                reduced_size_factor=4,
                dropout=0.2,
                fusion_strategy='attention',
                mlp_layers=3,  # Deeper for large models
                use_differences=True,
                use_rate=True,
                use_abs_time_mlp=True,  # Explicitly enable for advanced
                use_abs_mag_mlp=True
            )
        }
        return configs.get(model_size, configs['medium'])
    
    @staticmethod
    def get_transformer_config(embedding_size: int, memory_efficient: bool = False) -> EncoderConfig:
        """Get configuration optimized for transformer models."""
        return EncoderConfig(
            embedding_size=embedding_size,
            encoding_strategy='enhanced',
            reduced_size_factor=4 if memory_efficient else 2,
            dropout=0.1,
            fusion_strategy='mlp',
            mlp_layers=2 if memory_efficient else 3,  # Fewer layers for efficiency
            use_sinusoidal=True,
            use_conv_mag=True,
            use_differences=True,
            use_rate=True,
            use_band_embedding=True,
            use_abs_time_mlp=None,  # Auto-configure
            use_abs_mag_mlp=None
        )
    
    @staticmethod
    def get_convnet_config(embedding_size: int) -> EncoderConfig:
        """Get configuration optimized for convolutional networks."""
        return EncoderConfig(
            embedding_size=embedding_size,
            encoding_strategy='enhanced',
            reduced_size_factor=2,
            dropout=0.1,
            fusion_strategy='simple',
            mlp_layers=1,  # Simple for ConvNets
            use_sinusoidal=False,  # ConvNets might prefer direct embeddings
            use_conv_mag=True,
            use_differences=True,
            use_rate=False,  # Keep it simpler for ConvNets
            use_band_embedding=True,
            use_abs_time_mlp=False,  # Keep simple for ConvNets
            use_abs_mag_mlp=False
        )
    
    @staticmethod
    def get_ssl_config(embedding_size: int) -> EncoderConfig:
        """Get configuration optimized for self-supervised learning."""
        return EncoderConfig(
            embedding_size=embedding_size,
            encoding_strategy='advanced',
            reduced_size_factor=2,
            dropout=0.2,  # Higher dropout for SSL
            fusion_strategy='mlp',
            mlp_layers=4,  # Deeper for SSL complexity
            use_sinusoidal=True,
            use_conv_mag=True,
            use_differences=True,
            use_rate=True,
            use_band_embedding=True,
            use_abs_time_mlp=True,  # Enable for SSL
            use_abs_mag_mlp=True
        )
    
    @staticmethod
    def get_custom_config(features: Dict[str, bool], embedding_size: int = 256) -> EncoderConfig:
        """Create a custom configuration with specific features."""
        return EncoderConfig(
            embedding_size=embedding_size,
            encoding_strategy='custom',
            features_config=features,
            reduced_size_factor=2,
            dropout=0.1,
            fusion_strategy='mlp',
            mlp_layers=2,
            use_abs_time_mlp=None,  # Will be handled by features_config
            use_abs_mag_mlp=None
        )


def create_encoder_from_config(config: EncoderConfig, **kwargs):
    """Create a FlexiblePositionalEncoder from configuration."""
    # This would need to import the actual encoder class
    # from .flexible_positional_encoder import FlexiblePositionalEncoder
    
    encoder_kwargs = {
        'embedding_size': config.embedding_size,
        'encoding_strategy': config.encoding_strategy,
        'reduced_size_factor': config.reduced_size_factor,
        'dropout': config.dropout,
        'fusion_strategy': config.fusion_strategy,
        'mlp_layers': config.mlp_layers,
        'use_sinusoidal': config.use_sinusoidal,
        'use_conv_mag': config.use_conv_mag,
        'use_differences': config.use_differences,
        'use_rate': config.use_rate,
        'use_band_embedding': config.use_band_embedding,
        'use_abs_time_mlp': config.use_abs_time_mlp,
        'use_abs_mag_mlp': config.use_abs_mag_mlp,
        'num_bands': config.num_bands,
        'seq_length': config.seq_length,
        'features_config': config.features_config,
        **kwargs
    }
    
    # Return configuration dict that can be used to instantiate the encoder
    return encoder_kwargs


# Predefined configurations for common use cases
PRESET_CONFIGS = {
    'lightcurve_small': EncoderConfigManager.get_lightcurve_config('small'),
    'lightcurve_medium': EncoderConfigManager.get_lightcurve_config('medium'),
    'lightcurve_large': EncoderConfigManager.get_lightcurve_config('large'),
    
    'transformer_128': EncoderConfigManager.get_transformer_config(128),
    'transformer_256': EncoderConfigManager.get_transformer_config(256),
    'transformer_512': EncoderConfigManager.get_transformer_config(512),
    
    'convnet_128': EncoderConfigManager.get_convnet_config(128),
    'convnet_256': EncoderConfigManager.get_convnet_config(256),
    
    'ssl_256': EncoderConfigManager.get_ssl_config(256),
    'ssl_512': EncoderConfigManager.get_ssl_config(512),
    
    # Minimal configuration - only absolute MLP embeddings
    'minimal_128': EncoderConfig(
        embedding_size=128,
        encoding_strategy='custom',
        reduced_size_factor=2,
        dropout=0.1,
        fusion_strategy='simple',
        mlp_layers=1,
        use_sinusoidal=False,
        use_conv_mag=False,
        use_differences=False,
        use_rate=False,
        use_band_embedding=False,
        use_abs_time_mlp=True,
        use_abs_mag_mlp=True
    ),
}


def get_preset_config(preset_name: str) -> EncoderConfig:
    """Get a preset configuration by name."""
    if preset_name not in PRESET_CONFIGS:
        available = ', '.join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return PRESET_CONFIGS[preset_name]


def list_available_presets() -> list:
    """List all available preset configurations."""
    return list(PRESET_CONFIGS.keys())


# Example integration with your existing model
def integrate_with_atat_model(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example of how to integrate FlexiblePositionalEncoder with ATAT model.
    
    This shows how you might modify your model configuration to use
    the flexible encoder based on model parameters.
    """
    # Determine encoder config based on model parameters
    if model_config.get('use_lightcurve', True):
        embedding_size = model_config.get('lc_embedding_size', 256)
        
        # Choose strategy based on model complexity
        if model_config.get('model_type') == 'transformer':
            encoder_config = EncoderConfigManager.get_transformer_config(embedding_size)
        elif model_config.get('model_type') == 'convnet':
            encoder_config = EncoderConfigManager.get_convnet_config(embedding_size)
        elif model_config.get('ssl_training', False):
            encoder_config = EncoderConfigManager.get_ssl_config(embedding_size)
        else:
            # Default to lightcurve config
            model_size = 'large' if embedding_size >= 512 else 'medium' if embedding_size >= 256 else 'small'
            encoder_config = EncoderConfigManager.get_lightcurve_config(model_size)
        
        # Adjust for memory constraints
        if model_config.get('memory_efficient', False):
            encoder_config.reduced_size_factor = 4
            encoder_config.fusion_strategy = 'simple'
        
        # Convert to kwargs for instantiation
        encoder_kwargs = create_encoder_from_config(encoder_config)
        
        # Add to model config
        model_config['positional_encoder'] = encoder_kwargs
    
    return model_config


if __name__ == "__main__":
    # Example usage
    print("Available preset configurations:")
    for preset in list_available_presets():
        config = get_preset_config(preset)
        print(f"  {preset}: {config.encoding_strategy} strategy, size {config.embedding_size}")
    
    print("\nExample model integration:")
    example_model_config = {
        'use_lightcurve': True,
        'lc_embedding_size': 256,
        'model_type': 'transformer',
        'memory_efficient': True,
        'ssl_training': False
    }
    
    integrated_config = integrate_with_atat_model(example_model_config)
    print(f"Positional encoder config: {integrated_config['positional_encoder']}")
