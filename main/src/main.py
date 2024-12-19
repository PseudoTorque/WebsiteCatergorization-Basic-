"""
    This is the main file for the project.
"""

import os
import utils

#These are the hyperparameters for the model

BERT_MODEL_NAME = 'bert-base-uncased'
NUM_CLASSES = 16 #15 for large, 16 for small
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5
MODEL_PATH = 'data/model.pth'
DATA_PATH = 'data/WebsiteClassification(Small).csv'
MODE = 'small'

def main():

    train_dataloader, test_dataloader, class_mapping, tokenizer = utils.load_dataset(DATA_PATH, BERT_MODEL_NAME, BATCH_SIZE, MAX_LENGTH, MODE)

    if os.path.exists(MODEL_PATH):

        model, optimizer, scheduler, device = utils.initialize(BERT_MODEL_NAME, NUM_CLASSES, LEARNING_RATE, train_dataloader, NUM_EPOCHS)

        model = utils.load_model(MODEL_PATH, model)

    else:

        model, optimizer, scheduler, device = utils.initialize(BERT_MODEL_NAME, NUM_CLASSES, LEARNING_RATE, train_dataloader, NUM_EPOCHS)

    utils.train_model(model, train_dataloader, test_dataloader, NUM_EPOCHS, device, optimizer, scheduler)

    utils.save_model(model, MODEL_PATH)

    while True:

        url = input("Enter the URL to categorize: (Enter 'q' to quit)")

        if url == 'q':
            break

        category = utils.categorize_website(model, tokenizer, class_mapping, device, MAX_LENGTH, url)

        print(f"The URL is classified as: {category}")

if __name__ == "__main__":
    main()
