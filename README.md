# Automated Screening of Nail-Associated Melanoma Using KNN

Educational research prototype. **Not a medical device and not a diagnostic system.**
Flask web app + scikit-learn KNN on hand-crafted colour, texture and shape features.

> This project ships **without a dataset and without a trained model** on purpose.
> No accuracy numbers are included, because none exist until you train on your own data.

## 1. Setup (Windows, VS Code terminal)

```bat
cd nail-melanoma-screening
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
PowerShell blocking activation? Run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
(Or use Command Prompt instead of PowerShell.)

## 2. Add your dataset

```
dataset/
  melanoma/        <- melanoma-labelled nail images (.jpg .jpeg .png)
  non_melanoma/    <- all other nail images
  subjects.csv     <- OPTIONAL, see below
```
Make sure you have the right to use the images (licence / ethics approval) and that labels are trustworthy
(ideally biopsy- or clinician-confirmed).

**Optional `dataset/subjects.csv`** – if you know which patient each image came from, add:
```
filename,subject_id
melanoma/img001.jpg,P01
img002.jpg,P01
```
Then all images of one patient stay in the same split (no leakage). Without it, a stratified random split is used,
and images of the same patient may land in different splits (this can inflate results).

## 3. Audit the dataset
```bat
python -m src.dataset_loader
```
Prints and saves (`reports/dataset_audit.json`): images per class, unreadable files, duplicates, class imbalance.

## 4. Train and evaluate
```bat
python -m src.train_model
```
What it does: audit → features → 70/15/15 stratified split → scaler fitted on train only → grid search of K,
distance metric (euclidean, manhattan, chebyshev, cosine) and weighting on the **validation** set →
final model fitted on train → **one** evaluation on the test set.

Outputs: `models/knn_model.pkl`, `models/scaler.pkl`, `models/metadata.json`, `reports/*.json|csv`,
charts in `static/charts/`. To only redraw charts later: `python -m src.evaluate_model`.

Don't tune anything after looking at test results; if you change the model, you'd need new, untouched test data.

## 5. Run the web app
```bat
python app.py
```
Open http://127.0.0.1:5000 . Create an account and login. Home: drop a nail photo (it is analysed automatically) ->
Analysis page (5 steps: dataset, preprocessing, skin removal, features, KNN match) -> Result. Model Performance is in the top bar.
(Bootstrap loads from a CDN, so the browser needs internet access.)

## Project layout
```
app.py                    Flask app
src/config.py             paths + settings
src/dataset_loader.py     dataset scan + audit (corrupt/duplicate/imbalance)
src/preprocessing.py      resize 224x224, denoise, CLAHE, nail segmentation
src/feature_extraction.py 47 features (colour, GLCM texture, shape/edge)
src/train_model.py        split, scaling, KNN search, final model, saving
src/evaluate_model.py     metrics + charts
src/predict.py            loads model+scaler, predicts uploaded images
src/visualize.py          skin removal (GrabCut), enhanced nail, edge views (display only)
templates/ static/        web UI
notebooks/experimentation.ipynb   exploratory work (stops before the test set)
```

## Honest notes
- Skin-removal images are for display; the KNN features come from the same pipeline used in training, so no retraining is needed.
- Same `extract_from_image()` is used for training and for uploads, so the pipeline is consistent.
- Nail segmentation is a simple Otsu-based heuristic; if it fails, the whole frame is used and the app warns you.
- KNN "probabilities" are neighbour vote shares, not calibrated risk.
- Class imbalance: compare accuracy against the majority-class baseline shown on the Performance page; look at recall and specificity.
- Small datasets give very uncertain metrics. Report confidence intervals in any write-up.
- Uploaded images are stored in `uploads/` (private, login required) and auto-deleted after 24 hours.

## Troubleshooting
| Problem | Fix |
|---|---|
| `No trained model found` | Run `python -m src.train_model` first, then restart `python app.py` |
| `Missing folder: dataset\melanoma` | Create both class folders |
| A split contains only one class | You need more images per class |
| `ModuleNotFoundError` | Activate the venv, then `pip install -r requirements.txt` |
