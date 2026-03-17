#!/usr/bin/env python3
"""
Test script to verify the conv_time implementation in FlexiblePositionalEncoder
"""

def test_conv_time_parameter():
    """Test that the conv_time parameter is properly supported"""
    try:
        # Import the module
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from models.components.transformer.encoders.flexible_positional_encoder_simple import FlexiblePositionalEncoder
        
        # Test initialization with conv_time enabled
        encoder = FlexiblePositionalEncoder(
            embedding_size=128,
            seq_length=100,
            num_bands=2,
            use_conv_time=True,
            use_conv_mag=False,
            use_sinusoidal=False,
            use_time_diff=False,
            use_mag_diff=False,
            use_rate=False,
            use_band_embedding=False,
            use_abs_time_mlp=False,
            use_abs_mag_mlp=False
        )
        
        # Check that conv_time feature is enabled
        assert encoder.features['conv_time'] == True, "conv_time feature should be enabled"
        
        # Check that conv_time embedding layer is initialized
        assert hasattr(encoder, 'time_conv_emb_big'), "time_conv_emb_big should be initialized"
        
        print("✅ conv_time parameter test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def check_feature_configuration():
    """Check that features dictionary includes conv_time"""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from models.components.transformer.encoders.flexible_positional_encoder_simple import FlexiblePositionalEncoder
        
        # Test with conv_time disabled
        encoder1 = FlexiblePositionalEncoder(
            embedding_size=64,
            use_conv_time=False
        )
        assert encoder1.features['conv_time'] == False, "conv_time should be disabled"
        
        # Test with conv_time enabled
        encoder2 = FlexiblePositionalEncoder(
            embedding_size=64,
            use_conv_time=True
        )
        assert encoder2.features['conv_time'] == True, "conv_time should be enabled"
        
        print("✅ Feature configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Feature configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing conv_time implementation...")
    print()
    
    success = True
    success &= test_conv_time_parameter()
    success &= check_feature_configuration()
    
    print()
    if success:
        print("🎉 All tests passed! conv_time option has been successfully added.")
    else:
        print("❌ Some tests failed. Please check the implementation.")