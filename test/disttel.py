from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from google.cloud import bigquery
from torch.cuda.amp import autocast, GradScaler
import os
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

client = bigquery.Client(project="vz-it-np-j89v-dev-pol3co-0")

table = 'vz-it-pr-fjpv-vzdsdo-0.ds_work_tbls_no_gsam.nareab3_jan28_calls_parse'

query = f"""select *
    from {table} 
    """
#query
df = client.query(query).to_dataframe()
df.head()
df = df.dropna(subset = ["redacted_clean"])

query = f"""select *
    from vz-it-pr-fjpv-vzdsdo-0.ds_work_tbls_no_gsam.train_data_with_curated_lead_gen_billing_troubleshoot_15jan_2025 
    """
#query
df_labeled = client.query(query).to_dataframe()
df_labeled.head()
df_labeled = df_labeled.dropna(subset = ["redacted_clean"])

label_encoder = LabelEncoder()

df_labeled["mvp2_classes_mapped_modified"] = label_encoder.fit_transform(df_labeled["mvp2_classes_mapped_modified"])

id_to_label = {idx: label for idx, label in enumerate(label_encoder.classes_)}
label_to_id = {label: idx for idx, label in id_to_label.items()}

#  Resize Student Model Embeddings and Output Layer
def resize_student_model(teacher_tokenizer, student_model):
    teacher_vocab_size = len(teacher_tokenizer)
    student_vocab_size = student_model.config.vocab_size

    print(f"Teacher vocab size: {teacher_vocab_size}")
    print(f"Student vocab size before resizing: {student_vocab_size}")
    
  
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    student_model.config.pad_token_id = teacher_tokenizer.pad_token_id

 
    old_embeddings = student_model.get_input_embeddings()
    embedding_dim = old_embeddings.weight.shape[1]

    new_embeddings = nn.Embedding(teacher_vocab_size, embedding_dim)
    new_embeddings.weight.data[:old_embeddings.num_embeddings, :] = old_embeddings.weight.data
    student_model.set_input_embeddings(new_embeddings)

    
    old_lm_head = student_model.get_output_embeddings()
    new_lm_head = nn.Linear(embedding_dim, teacher_vocab_size, bias=False)
    new_lm_head.weight.data[:old_lm_head.out_features, :] = old_lm_head.weight.data
    student_model.set_output_embeddings(new_lm_head)

    student_model.config.vocab_size = teacher_vocab_size
    print(f"Student vocab size after resizing: {student_model.config.vocab_size}")

# Dataset 
class CallTranscriptDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.texts = dataframe["redacted_clean"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

# Distillation 
def distillation_training(teacher_model, student_model, tokenizer, dataset, device, temperature=2.0, epochs=20):
    batch_size = max(2 * torch.cuda.device_count(), 1)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    distillation_loss_fn = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler()
    
    teacher_model.gradient_checkpointing_enable()
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        teacher_model = torch.nn.DataParallel(teacher_model).to(device)
        student_model = torch.nn.DataParallel(student_model).to(device)
    else:
        teacher_model = teacher_model.to(device)
        student_model = student_model.to(device)

    for epoch in range(epochs):
        student_model.train()
        total_loss = 0

        for batch in dataloader:
            #print(f"Input IDs Shape: {batch['input_ids'].shape}")
            #print(f"Attention Mask Shape: {batch['attention_mask'].shape}")
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

        
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask=attention_mask).logits / temperature 

          
            #with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            student_logits = student_model(input_ids, attention_mask=attention_mask).logits / temperature 

            # Compute distillation loss
            teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
            student_probs = torch.nn.functional.log_softmax(student_logits, dim=-1)
            loss = torch.nn.functional.kl_div(student_probs, teacher_probs, reduction="batchmean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Distillation Loss = {total_loss / len(dataloader)}")

    student_model.save_pretrained("distilled_student_model")
    tokenizer.save_pretrained("distilled_student_model")
    print("Distillation completed. Student model saved.")

# Fine-Tuning the Student Model for Classification
def fine_tuning_for_classification(student_model, tokenizer, labeled_transcripts, device):
     
    #dataset = CallTranscriptClassificationDataset(labeled_transcripts, tokenizer)
    dataset = Dataset.from_pandas(df_labeled)
    
    # dataset = dataset.map(
    #     lambda example: {
    #         **tokenizer(
    #             example["redacted_clean"],
    #             max_length=128,
    #             truncation=True,
    #             padding="max_length",
    #         ),
    #         "label": [int(label) for label in example["mvp2_classes_mapped_modified"]],  
    #     },
    #     batched=True,
    # )
    
    def tokenize_and_format(batch):
        tokenized_inputs = tokenizer(
            batch["redacted_clean"],  # 
            max_length=128,
            truncation=True,
            padding="max_length",
        )
         
        tokenized_inputs["labels"] = [int(label) for label in batch["mvp2_classes_mapped_modified"]]
        return tokenized_inputs
    
    dataset = dataset.map(tokenize_and_format, batched=True)
    
    split_dataset = dataset.train_test_split(test_size=0.2, seed=32)
    num_labels = len(set(dataset["labels"]))

    # Classification model
    # classification_model = AutoModelForSequenceClassification.from_pretrained(
    #     "distilled_student_model", num_labels=num_labels
    # )
    student_model.config.pad_token_id = tokenizer.pad_token_id

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        fp16 = True,
        gradient_accumulation_steps = 2,
        save_steps = 1000
    )

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    results = trainer.evaluate()
    print(f"Evaluation Results: {results}")
    student_model.save_pretrained("fine_tuned_student_model")
    tokenizer.save_pretrained("fine_tuned_student_model")
    print("Fine-tuning completed. Fine-tuned model saved.")

teacher_model_name =   "EleutherAI/gpt-neo-1.3B" #EleutherAI/gpt-neox-20b" #"meta-llama/Llama-2-13b"    #"EleutherAI/gpt-neo-1.3B"
student_model_name =   "EleutherAI/gpt-neo-1.3B" #"bert-base-uncased"  #"bert-base-uncased"

teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
#teacher_model.gradient_checkpointing_enable()

student_model = AutoModelForCausalLM.from_pretrained(student_model_name)

# Resize student model embeddings
resize_student_model(teacher_tokenizer, student_model)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training!")
    teacher_model = torch.nn.DataParallel(teacher_model).to(device)
    student_model = torch.nn.DataParallel(student_model).to(device)
else:
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)


unlabeled_dataset = CallTranscriptDataset(df, teacher_tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distillation_training(teacher_model, student_model, teacher_tokenizer, unlabeled_dataset, device)

class CallTranscriptClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.texts = dataframe["redacted_clean"].tolist()
        self.labels = dataframe["mvp2_classes_mapped_modified"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


model_path  = "distilled_student_model"
student_tokenizer = AutoTokenizer.from_pretrained(model_path)
num_classes = len(label_encoder.classes_)
student_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = num_classes)
student_model.to(device)

# dataset = CallTranscriptClassificationDataset(df_labeled, student_tokenizer)\
# len(set(dataset.labels))

fine_tuning_for_classification(student_model, student_tokenizer, df_labeled, device)

dataset = Dataset.from_pandas(df_labeled)

# dataset = dataset.map(
#     lambda example: {
#         **tokenizer(
#             example["redacted_clean"],
#             max_length=128,
#             truncation=True,
#             padding="max_length",
#         ),
#         "label": [int(label) for label in example["mvp2_classes_mapped_modified"]],  
#     },
#     batched=True,
# )

def tokenize_and_format(batch):
    tokenized_inputs = student_tokenizer(
        batch["redacted_clean"],  # 
        max_length=128,
        truncation=True,
        padding="max_length",
    )

    tokenized_inputs["labels"] = [int(label) for label in batch["mvp2_classes_mapped_modified"]]
    return tokenized_inputs

dataset = dataset.map(tokenize_and_format, batched=True)

dataset[0]

model_name = "fine_tuned_student_model"   
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

model.eval()

id_to_label = {0: "Billing Issue", 1: "Technical Support", 2: "Sales Inquiry", 3: "General Inquiry"}


def classify_call_transcript(transcript):
    # Tokenize the input text
    inputs = tokenizer(
        transcript,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

     
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    
    predicted_label = id_to_label[predicted_class_id]
    return predicted_label

 
call_transcripts = [
    "Customer asked about a high bill this month.",
    "My internet is not working properly. Can you help?",
    "I would like to upgrade to a faster internet plan.",
    "How can I add another user to my current plan?",
]

# Perform inference on each transcript
for transcript in call_transcripts:
    prediction = classify_call_transcript(transcript)
    print(f"Transcript: {transcript}")
    print(f"Predicted Category: {prediction}")
    print("-" * 50)



