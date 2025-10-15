"""
Configuration management for Preterm Infant Pose Estimation
"""
import yaml
from easydict import EasyDict as edict


def get_default_config():
    """Get default configuration"""
    config = edict()
    
    # Model settings
    config.MODEL = edict()
    config.MODEL.NAME = 'pose_hrnet'
    config.MODEL.NUM_JOINTS = 13
    config.MODEL.IMAGE_SIZE = [256, 256]
    config.MODEL.HEATMAP_SIZE = [64, 64]
    config.MODEL.SIGMA = 2
    config.MODEL.FUSED_HEAD = True
    config.MODEL.PRETRAINED = ''
    
    # Loss settings
    config.LOSS = edict()
    config.LOSS.MORPH_WEIGHT = 0.1  # Weight for morphology loss
    config.LOSS.MORPH_LAMBDA = 1.0  # Lambda for variance constraint
    config.LOSS.REG_WEIGHT = 0.5    # Weight for regression loss
    config.LOSS.USE_TARGET_WEIGHT = True
    
    # Training settings
    config.TRAIN = edict()
    config.TRAIN.BATCH_SIZE = 32
    config.TRAIN.EPOCHS = 200
    config.TRAIN.LR = 0.001
    config.TRAIN.WEIGHT_DECAY = 0.0001
    config.TRAIN.NUM_WORKERS = 4
    config.TRAIN.VAL_INTERVAL = 5
    config.TRAIN.PRINT_FREQ = 100
    
    # Testing settings
    config.TEST = edict()
    config.TEST.BATCH_SIZE = 16
    config.TEST.NUM_WORKERS = 4
    config.TEST.FUSION_ALPHA = 0.5  # Weight for fused decoding
    config.TEST.NMS_THRESHOLD = 5.0
    config.TEST.CONF_THRESHOLD = 0.3
    
    # Data settings
    config.DATA = edict()
    config.DATA.DATA_DIR = './data'
    config.DATA.NUM_JOINTS = 13
    config.DATA.FLIP = True
    config.DATA.ROT_FACTOR = 30
    config.DATA.SCALE_FACTOR = 0.25
    
    # Output settings
    config.OUTPUT_DIR = './outputs'
    config.LOG_DIR = './logs'
    
    return config


def merge_config(config, args):
    """Merge configuration from file and command line arguments"""
    if hasattr(args, 'config') and args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            _update_config(config, yaml_config)
    
    return config


def _update_config(config, yaml_config):
    """Recursively update config dict"""
    for key, value in yaml_config.items():
        if key in config:
            if isinstance(value, dict) and isinstance(config[key], edict):
                _update_config(config[key], value)
            else:
                config[key] = value
        else:
            config[key] = value


def get_config(config_file=None):
    """
    Get configuration from file or use default
    
    Args:
        config_file: path to yaml config file
    """
    config = get_default_config()
    
    if config_file:
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
            _update_config(config, yaml_config)
    
    return config


def save_config(config, output_path):
    """Save configuration to yaml file"""
    # Convert edict to regular dict for yaml serialization
    config_dict = {}
    for key, value in config.items():
        if isinstance(value, edict):
            config_dict[key] = dict(value)
        else:
            config_dict[key] = value
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def print_config(config):
    """Pretty print configuration"""
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    
    def _print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, edict) or isinstance(value, dict):
                print(" " * indent + f"{key}:")
                _print_dict(value, indent + 2)
            else:
                print(" " * indent + f"{key}: {value}")
    
    _print_dict(config)
    print("=" * 80)


# Example configuration templates

def get_hrnet_w32_config():
    """HRNet-W32 configuration"""
    config = get_default_config()
    
    config.MODEL.NAME = 'pose_hrnet_w32'
    config.MODEL.NUM_JOINTS = 13
    config.MODEL.IMAGE_SIZE = [256, 256]
    config.MODEL.HEATMAP_SIZE = [64, 64]
    
    # HRNet specific
    config.MODEL.EXTRA = edict()
    config.MODEL.EXTRA.STAGE2 = edict()
    config.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
    config.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
    config.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
    config.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [32, 64]
    
    config.MODEL.EXTRA.STAGE3 = edict()
    config.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
    config.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
    config.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
    config.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [32, 64, 128]
    
    config.MODEL.EXTRA.STAGE4 = edict()
    config.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
    config.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
    config.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    config.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
    
    return config


def get_hrnet_w48_config():
    """HRNet-W48 configuration for higher accuracy"""
    config = get_hrnet_w32_config()
    
    config.MODEL.NAME = 'pose_hrnet_w48'
    
    # Wider network
    config.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
    config.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
    config.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
    
    # Larger batch size may be needed
    config.TRAIN.BATCH_SIZE = 24
    
    return config


def get_lightweight_config():
    """Lightweight configuration for faster inference"""
    config = get_default_config()
    
    config.MODEL.NAME = 'pose_mobilenet'
    config.MODEL.IMAGE_SIZE = [192, 192]
    config.MODEL.HEATMAP_SIZE = [48, 48]
    
    config.TRAIN.BATCH_SIZE = 64
    config.TRAIN.LR = 0.002
    
    return config


# Preterm infant specific configurations

def get_preemie_config():
    """
    Configuration optimized for preterm infant characteristics:
    - Small body sizes
    - Subtle motions
    - Low-frequency movements
    """
    config = get_default_config()
    
    # Adjust for small infant bodies
    config.MODEL.SIGMA = 1.5  # Smaller Gaussian for tiny keypoints
    config.MODEL.HEATMAP_SIZE = [128, 128]  # Higher resolution
    
    # Enhanced morphology loss for unstable movements
    config.LOSS.MORPH_WEIGHT = 0.15
    config.LOSS.MORPH_LAMBDA = 1.2
    
    # More fusion weight on regression for subtle motions
    config.TEST.FUSION_ALPHA = 0.4
    
    # Data augmentation adjusted for NICU environment
    config.DATA.ROT_FACTOR = 15  # Less rotation (infants are usually supine)
    config.DATA.SCALE_FACTOR = 0.15
    
    return config


if __name__ == '__main__':
    # Example usage
    config = get_default_config()
    print_config(config)
    
    # Save example configs
    import os
    os.makedirs('configs', exist_ok=True)
    
    save_config(get_default_config(), 'configs/default.yaml')
    save_config(get_hrnet_w32_config(), 'configs/hrnet_w32.yaml')
    save_config(get_preemie_config(), 'configs/preemie_optimized.yaml')
    
    print("Configuration files saved to configs/")
