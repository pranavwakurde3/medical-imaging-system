# Individual Reflection: AI-Assisted Medical Imaging System (AIML-02)

**Name:** Pranav Balasaheb Wakurde

## What I owned
I built this project from an empty folder: reading DICOM files, choosing and preprocessing a 600-patient subset of the RSNA dataset, training a DenseNet121 classifier with transfer learning, adding Grad-CAM heatmaps, running an Orthanc DICOM server, writing the inference service, building the React review viewer with an audit log, packaging everything in Docker, versioning the data with DVC, and training a simplified U-Net for segmentation.

## What my results were
(test set, 120 images, threshold 0.5)
- AUC: 0.826 (95% bootstrap interval 0.743 to 0.889)
- Sensitivity (pneumonia caught): 0.833 (50 of 60)
- Specificity (normal kept): 0.650 (39 of 60)
- Confusion matrix [tn fp fn tp]: 39 21 10 50
These come from one small test set, so they carry wide uncertainty.

The simplified segmentation model reached a test Dice of 0.487 on 60 pneumonia images.

These come from small test sets, so I treat them as rough evidence, not proof that the model works.

## What I learned
- **DICOM is more than a picture.** A DICOM file carries pixel data and metadata, and I had to normalize the pixels and check the patient fields before using the image.
- **Small data needs honest evaluation.** My first test run used 40 images, and its scores were meaningless, so I scaled to 600 images before reporting anything.
- **Transfer learning helps with little data.** I froze most of DenseNet121 and trained only the last block and a new output layer.
- **Explanations are not proof.** Grad-CAM shows where the model looked, not that it is right, so I checked whether the heat sat inside the lungs.
- **A human stays in the loop.** The viewer shows the AI score next to the scan, and every Approve or Reject is saved with the reviewer, the AI score and the time.
- **Limits and bias must be written down.** I wrote a model card with the limits and possible bias, and a clear "for education only, not clinical use" notice.
- **Debugging infrastructure.** SQLite raised a disk I/O error when two containers shared a database on a Windows folder, and I fixed it by moving the database into a Docker volume.

## What was hard
My internet was slow, so large downloads and Docker builds took hours. I learned to test the whole pipeline on a few images first, and to check each layer (image, container, database, viewer) separately.

## What I did not finish or simplified
- Segmentation uses a 2D U-Net in plain PyTorch trained on rectangle masks made from bounding boxes, not the 3D U-Net in MONAI named in the project.
- DVC stores data in a local folder, not in cloud storage.
- The viewer has no login, and Orthanc authentication is off.
- The model was trained on a CPU, so it is far from clinical grade.

## What I would do next
- Train on more images, with a GPU, and measure results on a separate hospital's data.
- Test performance by age, sex and scan type to look for bias.
- Add login and access control to the viewer and to Orthanc.
- Use real segmentation masks and the 3D U-Net from MONAI.
