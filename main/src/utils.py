"""

Utility functions for the WebsiteCatergorization module.

"""

from csv import reader

from string import punctuation
from requests import get
from bs4 import BeautifulSoup

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def get_bytes_from_url(url: str, timeout: int = 10) -> bytes:

    """
    Get the bytes from a URL.

    Args:
        url (str): The URL to get the bytes from.
        timeout (int, optional): The timeout for the request. Defaults to 10.

    Returns:
        bytes: The bytes from the URL.
    """

    response = get(url, timeout=timeout)

    response.raise_for_status()

    return response.content

def get_hyperlinks_from_page_bytes(page_bytes: bytes) -> list[str]:

    """
    Get the hyperlinks from a page.

    Args:
        page_bytes (bytes): The bytes of the page.

    Returns:
        list[str]: The hyperlinks from the page.
    """

    soup = BeautifulSoup(page_bytes, "html.parser")

    hyperlinks = []

    link: BeautifulSoup = None

    for link in soup.find_all("a"):
        hyperlinks.append(link.get("href", default=None))

    return [i for i in hyperlinks if i is not None]

def clean_html(page_bytes: bytes) -> str:

    """
    Clean the HTML of a page (in bytes).

    Args:
        page_bytes (bytes): The bytes of the page.

    Returns:
        str: The cleaned HTML.
    """

    soup = BeautifulSoup(page_bytes, "html.parser")

    for data in soup(['style', 'script', 'code', 'a']):

        data.decompose()

    return ' '.join(soup.stripped_strings)

test = get_bytes_from_url("https://pulse.zerodha.com")

def clean_text(text: str) -> str:

    """
    Clean the text of a page (convert to lowercase, remove stopwords, remove punctuation).

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """

    #convert text to lowercase
    text = text.lower()

    #remove newlines
    text = text.replace("\n", " ")

    #remove whitespace
    text = [word for word in text.split() if word.strip()]

    #remove punctuation
    translator = str.maketrans('', '', punctuation+"—")
    text = " ".join(text).translate(translator)

    return text

class WebsiteCatergorizationDataset(Dataset):

    """
    Dataset for the WebsiteCatergorization module.

    Args:
        Dataset (torch.utils.data.Dataset): The base class for all datasets.
    """

    def __init__(
                self,
                labels: list[int],
                tokenizer: BertTokenizer,
                urls: list[str] = None,
                texts: list[str] = None,
                max_length: int = 512
                ):

        self.urls = urls

        self.texts = texts

        self.labels = labels

        self.tokenizer = tokenizer

        self.max_length = max_length

    def __len__(self):

        return len(self.urls) if self.urls is not None else len(self.texts)

    def __getitem__(self, index: int):

        if self.urls is not None:
            url = self.urls[index]

            page_bytes = get_bytes_from_url(url)

            text = clean_text(clean_html(page_bytes))

        elif self.texts is not None:
            text = self.texts[index]

        else:
            raise ValueError("Either URLs or Texts must be provided! ")

        encoding = self.tokenizer(
                                text,
                                return_tensors="pt",
                                truncation=True,
                                padding="max_length",
                                max_length=self.max_length
        )

        label = self.labels[index]

        out = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label)
        }

        return out

class BERTClassifier(nn.Module):

    """
    BERT Classifier for the WebsiteCatergorization module.

    Args:
        nn.Module (torch.nn.Module): The base class for all neural network modules.
    """

    def __init__(self, bert_model_name: str, num_classes: int):

        super(BERTClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        """
        Forward pass for the BERTClassifier.

        Args:
            input_ids (torch.Tensor): The input IDs.
            attention_mask (torch.Tensor): The attention mask.

        Returns:
            torch.Tensor: The logits.
        """

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.pooler_output

        x = self.dropout(pooled_output)

        logits = self.fc(x)

        return logits

def train(model: nn.Module, data_loader: DataLoader, optimizer: any, scheduler: any, device: str):

    """
    Train the model.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): The data loader.
        optimizer (any): The optimizer.
        scheduler (any): The scheduler.
        device (str): The device to train on.
    """

    model.train()

    for batch in data_loader:

        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)

        attention_mask = batch['attention_mask'].to(device)

        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = nn.CrossEntropyLoss()(outputs, labels)

        loss.backward()

        optimizer.step()

        scheduler.step()

def evaluate(model: nn.Module, data_loader: DataLoader, device: str):

    """
    Evaluate the model.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): The data loader.
        device (str): The device to evaluate on.
    """

    model.eval()

    predictions = []

    actual_labels = []

    with torch.no_grad():

        for batch in data_loader:

            input_ids = batch['input_ids'].to(device)

            attention_mask = batch['attention_mask'].to(device)

            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().tolist())

            actual_labels.extend(labels.cpu().tolist())

    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def categorize_website(model: nn.Module, tokenizer: BertTokenizer, class_mapping: dict, device: str, max_length: int = 512, url: str = None):

    """
    Categorize a website.

    Args:
        model (nn.Module): The model to use for prediction.
        tokenizer (BertTokenizer): The tokenizer to use for encoding the text.
        class_mapping (dict): The class mapping.
        device (str): The device to use for prediction.
        max_length (int, optional): The maximum length of the text. Defaults to 128.
        url (str, optional): The URL to categorize. Defaults to None.
    """

    model.eval()

    page_bytes = get_bytes_from_url(url)

    to_predict = clean_text(clean_html(page_bytes))
    
    encoding = tokenizer(to_predict, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)

    input_ids = encoding['input_ids'].to(device)

    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
            
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)

        return class_mapping[preds.item()]

def load_dataset(data_path: str, bert_model_name: str, batch_size: int, max_length: int, mode:str = "small"):

    """
    Load the dataset.

    Args:
        data_path (str): The path to the dataset.
        mode (str, optional): The mode to load the dataset in. Defaults to "small".
    """

    #open csv file and read it
    with open(data_path, mode='r', encoding='utf-8') as file:

        temp_data = reader(file)

        data = [row for row in temp_data]

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    if mode == "small":

        data = [{"website_url": row[1], "cleaned_text": row[2], "category": row[3]} for row in data[1:]]

        class_mapping = {label: i for i, label in enumerate(set(row["category"] for row in data))}

        labels = [class_mapping[row["category"]] for row in data]

        data = [row["cleaned_text"] for row in data]

        train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2, random_state=43)

        train_dataset = WebsiteCatergorizationDataset(labels=train_y, tokenizer=tokenizer, texts=train_x, max_length=max_length)

        test_dataset = WebsiteCatergorizationDataset(labels=test_y, tokenizer=tokenizer, texts=test_x, max_length=max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif mode == "large":

        data = [{"website_url": row[1], "category": row[2]} for row in data]

        class_mapping = {label: i for i, label in enumerate(set(row["category"] for row in data))}

        labels = [class_mapping[row["category"]] for row in data]

        data = [row["website_url"] for row in data]

        train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2, random_state=43)

        train_dataset = WebsiteCatergorizationDataset(labels=train_y, tokenizer=tokenizer, urls=train_x, max_length=max_length)

        test_dataset = WebsiteCatergorizationDataset(labels=test_y, tokenizer=tokenizer, urls=test_x, max_length=max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    else:

        raise ValueError("Invalid mode! ")

    class_mapping = {j: i for i, j in class_mapping.items()}

    return train_loader, test_loader, class_mapping, tokenizer

def initialize(bert_model_name: str, num_classes: int, learning_rate: float, train_dataloader: DataLoader, num_epochs: int):

    """
    Initialize the model, optimizer and scheduler.

    Args:
        bert_model_name (str): The name of the BERT model.
        num_classes (int): The number of classes.
        learning_rate (float): The learning rate.
        train_dataloader (DataLoader): The training data loader.
        num_epochs (int): The number of epochs.

    Returns:
        model (nn.Module): The model.
        optimizer (any): The optimizer.
        scheduler (any): The scheduler.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BERTClassifier(bert_model_name, num_classes).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_dataloader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    return model, optimizer, scheduler, device

def train_model(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int, device: str, optimizer: any, scheduler: any):

    """
    Train the model.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): The training data loader.
        val_dataloader (DataLoader): The validation data loader.
        num_epochs (int): The number of epochs.
        device (str): The device to train on.
        optimizer (any): The optimizer.
        scheduler (any): The scheduler.
    """

    for epoch in range(num_epochs):

        print(f"Epoch {epoch + 1}/{num_epochs}")

        train(model, train_dataloader, optimizer, scheduler, device)

        accuracy, report = evaluate(model, val_dataloader, device)

        print(f"Validation Accuracy: {accuracy:.4f}")

        print(report)

def save_model(model: nn.Module, model_path: str):

    """
    Save the model.

    Args:
        model (nn.Module): The model to save.
        model_path (str): The path to save the model.
    """

    torch.save(model.state_dict(), model_path)

def load_model(model_path: str, model: nn.Module):

    """
    Load the model.

    Args:
        model_path (str): The path to load the model.
        model (nn.Module): The model to load.
    """

    model.load_state_dict(torch.load(model_path))

    return model



