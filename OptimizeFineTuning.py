import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    Trainer, DataCollatorForLanguageModeling, TextDataset
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training

model_name = "JackFram/llama-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')


file_paths = [
    #"PythonApplication1/out.txt",
    "meta2.txt",
]

datasets = []
for file_path in file_paths:
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128  # Adjust according to your data size
    )
    datasets.append(dataset)

if len(datasets) > 1:
    concatenated_dataset = torch.utils.data.ConcatDataset(datasets)
else:
    concatenated_dataset = datasets[0]

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)




#4 bit configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", #qlora formatı
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
peft_config = LoraConfig(
    lora_alpha=10,
    #10% dropout
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load base moodel
model = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=bnb_config,
    device_map={"": 0}
)

model.config.use_cache = False
model.config.pretraining_tp = 1
training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,#önerilen epoch sayısı
        per_device_train_batch_size=4,# her adıma
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",#train etmek istiyorum
        eval_steps=1000,
        logging_steps=25,
        optim="paged_adamw_8bit",#adamw ihtiyacımız olan
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=10,
        report_to="tensorboard",
)



trainer = Trainer(
    model=model,
    train_dataset=concatenated_dataset,
    eval_dataset=concatenated_dataset,#ayrı bir veri setimiz yok aynı dataset
    peft_config=peft_config,
    data_collator=data_collator,
    max_seq_length=512,  # daha fazla vram olsa daha iyi olurdu
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()
# Save the fine-tuned model
model.save_pretrained("fine_tuned2_model")
tokenizer.save_pretrained("fine_tuned2_model")