import optuna
from omegaconf import DictConfig


def validate_flexible_encoder_constraints(cfg: DictConfig) -> None:
    """
    Validate flexible encoder constraints during Optuna optimization.
    Add this function call at the beginning of your training function.
    
    Args:
        cfg: Hydra configuration
        
    Raises:
        optuna.TrialPruned: If constraints are violated
    """
    # Check if we're in an Optuna trial
    try:
        trial = optuna.integration.get_current_trial()
        if trial is None:
            return  # Not in an Optuna trial, skip validation
    except:
        return  # Not in an Optuna context, skip validation
    
    # Get encoder configuration
    encoder_cfg = cfg.model.lc_net.time_encoder
    
    # Extract boolean parameters
    use_sinusoidal = getattr(encoder_cfg, 'use_sinusoidal', False)
    use_conv_mag = getattr(encoder_cfg, 'use_conv_mag', False)
    use_mag_diff = getattr(encoder_cfg, 'use_mag_diff', False)
    use_time_diff = getattr(encoder_cfg, 'use_time_diff', False)
    use_rate = getattr(encoder_cfg, 'use_rate', False)
    use_band_embedding = getattr(encoder_cfg, 'use_band_embedding', False)
    use_abs_time_mlp = getattr(encoder_cfg, 'use_abs_time_mlp', False)
    use_abs_mag_mlp = getattr(encoder_cfg, 'use_abs_mag_mlp', False)
    
    # Apply constraints
    # Constraint 1: Cannot use both mag_diff and time_diff
    if use_mag_diff and use_time_diff:
        raise optuna.TrialPruned()
    
    # Constraint 2: Must have at least one magnitude and one time feature
    mag_active = any([use_conv_mag, use_mag_diff, use_abs_mag_mlp])
    time_active = any([use_sinusoidal, use_time_diff, use_abs_time_mlp])
    
    # Constraint 3: If using rate, both mag and time must be active
    if use_rate:
        mag_active = True
        time_active = True
    
    # Constraint 4: Must have both magnitude and time features active
    if not mag_active or not time_active:
        raise optuna.TrialPruned()


class FlexibleEncoderObjective:
    """
    Alternative objective class for more complex Optuna optimization.
    Use this if you want to handle the optimization logic manually.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self, trial: optuna.trial.Trial):
        # Suggest booleans
        use_sinusoidal = trial.suggest_categorical('use_sinusoidal', [True, False])
        use_conv_mag = trial.suggest_categorical('use_conv_mag', [True, False])
        use_mag_diff = trial.suggest_categorical('use_mag_diff', [True, False])
        use_time_diff = trial.suggest_categorical('use_time_diff', [True, False])
        use_rate = trial.suggest_categorical('use_rate', [True, False])
        use_band_embedding = trial.suggest_categorical('use_band_embedding', [True, False])
        use_abs_time_mlp = trial.suggest_categorical('use_abs_time_mlp', [True, False])
        use_abs_mag_mlp = trial.suggest_categorical('use_abs_mag_mlp', [True, False])

        # Enforce constraints
        if use_mag_diff and use_time_diff:
            raise optuna.TrialPruned()
        mag_active = any([use_conv_mag, use_mag_diff, use_abs_mag_mlp])
        time_active = any([use_sinusoidal, use_time_diff, use_abs_time_mlp])
        if use_rate:
            mag_active = True
            time_active = True
        if not mag_active or not time_active:
            raise optuna.TrialPruned()

        # Update config for this trial
        self.cfg.model.lc_net.time_encoder.use_sinusoidal = use_sinusoidal
        self.cfg.model.lc_net.time_encoder.use_conv_mag = use_conv_mag
        self.cfg.model.lc_net.time_encoder.use_mag_diff = use_mag_diff
        self.cfg.model.lc_net.time_encoder.use_time_diff = use_time_diff
        self.cfg.model.lc_net.time_encoder.use_rate = use_rate
        self.cfg.model.lc_net.time_encoder.use_band_embedding = use_band_embedding
        self.cfg.model.lc_net.time_encoder.use_abs_time_mlp = use_abs_time_mlp
        self.cfg.model.lc_net.time_encoder.use_abs_mag_mlp = use_abs_mag_mlp

        # Initialize model and datamodule
        import hydra
        model = hydra.utils.instantiate(self.cfg.model)
        datamodule = hydra.utils.instantiate(self.cfg.data)

        from pytorch_lightning import Trainer
        trainer = Trainer(**self.cfg.trainer)

        # Train model
        trainer.fit(model, datamodule=datamodule)

        # Validate and get metric
        metrics = trainer.validate(model, datamodule=datamodule, verbose=False)
        val_f1 = metrics[0]['val/f1']  # Adjust according to your metric name

        return val_f1