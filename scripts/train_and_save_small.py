from src.data import load_california
from src.features import df_to_Xy, get_numeric_features, build_preprocessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib, pathlib, os

print('loading data...')
df = load_california(sample_n=200)
X, y = df_to_Xy(df)
print('building pipeline...')
num_feats = get_numeric_features(X)
pre = build_preprocessor(numeric_features=num_feats)
pipe = Pipeline([('pre', pre), ('model', RandomForestRegressor(n_estimators=5, random_state=42, n_jobs=1))])
print('fitting...')
pipe.fit(X, y)
print('saving...')
path = pathlib.Path('models'); path.mkdir(exist_ok=True)
joblib.dump(pipe, path / 'best_pipeline.pkl', compress=3)
print('wrote', path / 'best_pipeline.pkl', 'size=', os.path.getsize(str(path / 'best_pipeline.pkl')))
