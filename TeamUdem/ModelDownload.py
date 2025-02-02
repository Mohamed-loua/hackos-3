#%%
from transformers import AutoModelForCausalLM, AutoTokenizer,  TrainingArguments, Trainer

model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

#%%

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)


# %%

save_folder_name = './model/my_model'

model.save_pretrained(save_folder_name)
tokenizer.save_pretrained(save_folder_name)
# %%

import pandas as pd
# Data visualization task 

df = pd.read_csv('../data/actual/dataset.csv', delimiter='|')

# %%



tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load your dataset for Stage 1
# %%
# Format the dataset for Stage 1
df_stage1 = df[["input", "error_type", "severity"]].copy()
df_stage1["output"] = "Error Type: " + df_stage1["error_type"] + ", Severity: " + df_stage1["severity"]

# %%
# Convert to Hugging Face Dataset format
from datasets import Dataset

dataset_stage1 = Dataset.from_pandas(df_stage1[["input", "output"]])

# %%
#Format the dataset for Stage 2
df_stage2 = df[["input", "error_type", "severity", "description", "solution"]].copy()
df_stage2["output"] = (
    "Description: " + df_stage2["description"] + " Solution: " + df_stage2["solution"]
)

# Convert to Hugging Face Dataset format
dataset_stage2 = Dataset.from_pandas(df_stage2[["input", "error_type", "severity", "output"]])
# %%
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        f"Input: {examples['input']}\nOutput: {examples['output']}",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

#%%
# Tokenize the dataset1

tokenized_dataset_stage1 = dataset_stage1.map(tokenize_function, batched=True)

#%%
# Define training arguments
training_args = TrainingArguments(
    output_dir="./stage1_results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

#%%
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_stage1,
)

# Fine-tune the model
trainer.train()