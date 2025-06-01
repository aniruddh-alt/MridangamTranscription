# Running Mridangam Transcription on Kaggle

This document explains how to use the packaged mridangam transcription model on Kaggle.

## Prerequisites

- A Kaggle account
- Basic familiarity with Kaggle notebooks

## Dataset Adaptation Notes

- The dataset has been modified for Kaggle compatibility:
  - Folders and filenames containing `#` have been renamed to use `sharp` instead (e.g., `C#` → `Csharp`, `D#` → `Dsharp`)
  - The code has been updated to handle this change transparently, so the model still uses the original stroke labels

## Steps to Run on Kaggle

1. **Upload the Dataset**
   - Go to Kaggle and click on "Datasets" in the top menu
   - Click "New Dataset"
   - Upload the `mridangam_transcription_kaggle.zip` file
   - Set the dataset title to "mridangam-transcription"
   - Set the dataset subtitle to "Mridangam stroke classification dataset"
   - Click "Create"

2. **Create a New Notebook**
   - Go to the "Code" tab of your dataset
   - Click "New Notebook"
   - Select the GPU accelerator (P100 or T4) from the notebook settings

3. **Import the Provided Notebook**
   - Delete the default code cell
   - Click "File" → "Import Notebook"
   - Browse for the `kaggle_notebook.ipynb` file from the `code` directory in the dataset
   - Alternatively, you can copy-paste the content from that file

4. **Run the Notebook**
   - Click "Run All" to execute all cells
   - The notebook will:
     - Set up the necessary imports
     - Process the dataset
     - Train the model
     - Evaluate performance
     - Save model checkpoints

## Customization Options

You can modify these parameters in the notebook to experiment:

- **Batch size**: Adjust based on GPU memory (default: 32)
- **Learning rate**: Tune for better convergence (default: 0.001)
- **Number of epochs**: Increase for potentially better results (default: 50)
- **Architecture**: Change between 'cnn', 'cnn_rnn', 'cnn_lstm', etc.

## Using GPU Efficiently

Kaggle provides P100 or T4 GPUs with limited runtime (maximum 30 hours per week). To make the most of your GPU quota:

1. Set `pin_memory=True` in DataLoader for faster data transfer to GPU
2. Use `torch.cuda.empty_cache()` between training runs
3. Implement early stopping to avoid wasting compute time
4. Save model checkpoints regularly

## Troubleshooting

- **Memory errors**: Reduce batch size
- **Slow processing**: Increase num_workers (up to 4)
- **Missing dependencies**: Add any missing dependencies with `!pip install package_name`

## Viewing Results

Your trained model will be saved as `mridangam_model_final.pth`. You can visualize training progress using Kaggle's built-in charts or by adding your own visualization code.
