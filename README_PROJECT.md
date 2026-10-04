# AI/ML Model for Predicting Reusable Battery Life

This project predicts the Remaining Useful Life (RUL) of used lithium-ion batteries and provides a second-life screening dashboard for applications such as stationary solar storage, emergency lighting, and portable power systems.

## Architecture

Battery measurements -> validation -> Random Forest RUL regression -> predicted RUL -> condition/second-life screening -> Streamlit dashboard.

## Structure

- `app.py` - Streamlit dashboard
- `train_model.py` - reproducible training CLI
- `src/pipeline.py` - validation and Random Forest training
- `tests/test_pipeline.py` - schema/validation tests
- `requirements.txt` - runtime and test dependencies
- `rul_model(1).pkl` - existing trained model artifact

## Model inputs

`cycle`, `voltage`, `current`, `temperature`, `capacity`, `resistance`, `charge_time`, `discharge_time`

Training target: `rul` in cycles.

## Run

`pip install -r requirements.txt`

`streamlit run app.py`

## Retrain

`python train_model.py --data data/processed/battery_rul.csv --output rul_model.pkl`

## Important

RUL prediction is a screening aid, not a battery safety certification. Real second-life deployment requires appropriate electrical, thermal, physical, and safety testing.
