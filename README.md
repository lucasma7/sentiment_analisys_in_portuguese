## SENTIMENT ANALISYS IN PORTUGUESE

### SENTIMENT ANALYSIS OF TEXTS IN PORTUGUESE USING NATURAL LANGUAGE PROCESSING TECHNIQUES
### Running locally
Using some Linux distro and make sure you have Python 3 installed.

Clone the project:

```bash 
git clone https://github.com/lucasma7/sentiment_analisys_in_portuguese.git 
```
Access the project directory:

```bash
cd sentiment_analisys_in_portuguese
```
Add a models for inference in the models file, you can run the sentiment_analysis.ipynb notebook in a colab environment and it will automatically load the dataset, train the model and save the best model locally in the downloads folder.

Creating a virtual environment (for the example we use the location directory parameter as .venv):

```bash
python3 -m venv .venv
```
or conda

```bash
conda create -n venv python=3.10
```


Activating the virtual environment:

```bash
source .venv/bin/activate
```
or conda

```bash
conda activate venv
```

Install all required packages specified in requirements.txt:
```bash
  pip install -r requirements.txt
```
Use the following command to run the predicts sentiments:
```bash
python -m test_inference
```

### References
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

Dataset: [Kaggle](https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis)
