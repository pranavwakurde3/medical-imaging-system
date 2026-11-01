# Model Card: Chest X-ray Pneumonia Classifier

**FOR EDUCATION ONLY. NOT FOR CLINICAL USE.**

## Model
DenseNet121 pre-trained on ImageNet. Only the last dense block and a new output layer were trained (transfer learning). Input: chest X-ray resized to 224x224.

## Data
RSNA Pneumonia Detection Challenge (Kaggle), DICOM format. A random subset of 600 patients: 300 with pneumonia labels and 300 without. Split: 420 train, 60 validation, 120 test, stratified by label.

## Measured results (test set, 120 images, threshold 0.5)
- AUC: 0.826 (95% bootstrap interval 0.743 to 0.889)
- Sensitivity (pneumonia caught): 0.833 (50 of 60)
- Specificity (normal kept): 0.650 (39 of 60)
- Confusion matrix [tn fp fn tp]: 39 21 10 50
These come from one small test set, so they carry wide uncertainty.

## Limits
- Trained on a small subset using a CPU, so performance is far below a clinical-grade system.
- The classes were balanced to 50/50 for training. Real pneumonia prevalence is much lower, so precision in real use would be lower than shown here.
- One dataset, one source. The model has not been tested on other hospitals, scanners or protocols.
- Labels come from the challenge annotations, not from my own radiology review.
- Grad-CAM shows where the model looked, not that it is correct. Heat on image borders, markers or tubes would signal shortcut learning.
- The viewer runs without login, and Orthanc authentication is off. This is a local demo only.

## Possible bias
- The data comes from a single source population, so age, sex, body size and equipment differences are not represented evenly. Subgroup performance was not measured.
- Portable and bedside X-rays often show more severe cases, so the model may learn the scan type instead of the disease.
- I did not audit performance by sex or age because the preprocessed images no longer carry that information.

## Intended use
Learning how a DICOM, inference and review workflow fits together. Not for diagnosis, triage or any decision about a patient.
