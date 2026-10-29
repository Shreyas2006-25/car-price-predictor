# CarWorth – Used Car Price Predictor

ML model + GUI that estimates second-hand car prices (INR) from brand, model, year, kilometres, fuel,
transmission, seller type and ownership. Trained on the real **CarDekho** dataset (Kaggle:
"Vehicle dataset from CardekhoCar"), 4,340 listings -> 3,574 after removing duplicates/outliers.

## Run
```
pip install -r requirements.txt
python train.py          # optional: a trained model is already in ./model
streamlit run app.py     # opens the GUI in your browser
```

## How it works
- `train.py` – cleans data, builds features (car age, log-km, one-hot / target-encoded categories), predicts
  log(price), compares Ridge / Random Forest / Gradient Boosting / Hist-GB with 5-fold CV, saves the best.
- `app.py`   – Streamlit GUI: live price estimate with range, depreciation + mileage curves, similar listings,
  market explorer, and model-performance tab.
- Result: test R² ≈ 0.76, mean error ≈ 25 %. The dataset lacks engine/power/condition, which caps accuracy.

## Using your own Kaggle CSV
Put it in `data/` with columns: Brand, Model, Year, Selling_Price, KM_Driven, Fuel, Seller_Type,
Transmission, Owner, then `python train.py --data data/yourfile.csv`.
