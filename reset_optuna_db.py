import optuna
import os

# Remove existing database
db_path = "optuna_study.db"
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Removed existing database: {db_path}")

# Create a fresh study to initialize the database
study = optuna.create_study(
    study_name="EncoderAblationStudy",
    storage=f"sqlite:///{db_path}",
    direction="maximize",
    load_if_exists=False
)

print(f"Created fresh database: {db_path}")
print(f"Study name: {study.study_name}")