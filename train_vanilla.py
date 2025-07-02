from datasets import load_dataset, Image  # Add this import

# === Configurable Parameters ===
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
STEP_SIZE = 5
LR_GAMMA = 0.5
from transformers import CLIPProcessor, CLIPVisionModel
from transformers import GPT2LMHeadModel
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from tqdm import tqdm
import warnings
import wandb
from PIL import ImageDraw, ImageFont
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values`")


full_dataset = load_dataset("nlphuji/flickr30k", split="test[:50%]").cast_column("image", Image())
train_size = int(0.8 * len(full_dataset))
eval_size = len(full_dataset) - train_size
dataset = full_dataset.train_test_split(train_size=train_size, test_size=eval_size, shuffle=True, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()  # freeze
clip_model.requires_grad_(False)


class MiniQFormer(nn.Module):
    def __init__(self, vision_width=512, num_query_tokens=32, hidden_dim=512, num_layers=2, num_heads=8):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(vision_width, hidden_dim)

    def forward(self, vision_embeds):
        # vision_embeds: [batch, seq, vision_width]
        batch_size = vision_embeds.size(0)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        vision_proj = self.proj(vision_embeds)
        # Attend queries to vision features
        x = torch.cat([queries, vision_proj], dim=1)
        x = self.transformer(x)
        # Only return the queries part
        return x[:, :queries.size(1), :]


gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# --- Collator, DataLoader, Training, Evaluation ---
from torch.utils.data import DataLoader
from torchvision import transforms
import random

# Collator to tokenize captions and preprocess images
class FlickrCollator:
    def __init__(self, processor, tokenizer, max_length=32):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        captions = [item["caption"][0] for item in batch]  # just use the first caption
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        tokens = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length, return_attention_mask=True)
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "caption_lengths": [len(ids) for ids in tokens.input_ids]
        }

collator = FlickrCollator(clip_processor, gpt2_tokenizer)
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)



# Simple training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
q_former = MiniQFormer(vision_width=768, hidden_dim=768).to(device)
gpt2_model = gpt2_model.to(device)
optimizer = torch.optim.Adam(list(q_former.parameters()) + list(gpt2_model.parameters()), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_GAMMA)  # decay lr by LR_GAMMA every STEP_SIZE epochs
gpt2_model.train()

def train():
    import os
    wandb.init(project="image-captioning", name="vanilla_transformer", config={
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE
    })

    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}")
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            with torch.no_grad():
                vision_outputs = clip_model(batch["pixel_values"].to(device)).last_hidden_state
            q_output = q_former(vision_outputs) #[batch_size, 32, 512]
            q_output = q_output.to(dtype=torch.float32)

            # Add start token (like "A photo of") to guide generation
            start_token = gpt2_tokenizer.encode("A photo of", return_tensors="pt").to(device)
            start_embeds = gpt2_model.get_input_embeddings()(start_token).repeat(q_output.size(0), 1, 1)

            # Concatenate start + q_output to form visual prompt
            visual_prompt = torch.cat([start_embeds, q_output], dim=1)

            # Get token embeddings of the target captions
            caption_embeddings = gpt2_model.transformer.wte(batch["input_ids"].to(device))

            # Combine visual prompt + caption
            inputs_embeds = torch.cat([visual_prompt, caption_embeddings], dim=1)

            # Pad the labels accordingly
            pad_len = visual_prompt.size(1)
            labels = torch.cat([
                torch.full((batch["input_ids"].size(0), pad_len), -100, dtype=torch.long).to(device),  # ignore loss on visual prompt
                batch["input_ids"].to(device)
            ], dim=1)

            outputs = gpt2_model(
                inputs_embeds=inputs_embeds,
                labels=labels
            )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

        # Show predictions on 3 samples after each epoch
        gpt2_model.eval()
        q_former.eval()
        print("\nSample predictions after this epoch:")
        for i in range(3):
            item = random.choice(eval_dataset)
            image = item["image"]
            reference = item["caption"][0]

            pixel_values = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                vision_outputs = clip_model(pixel_values).last_hidden_state

                q_output = q_former(vision_outputs).to(dtype=torch.float32)

                start_token = gpt2_tokenizer.encode("A photo of", return_tensors="pt").to(device)
                start_embeds = gpt2_model.get_input_embeddings()(start_token).repeat(q_output.size(0), 1, 1)
                visual_prompt = torch.cat([start_embeds, q_output], dim=1)

                # Print mean and std of visual prompt
                mean = visual_prompt.mean().item()
                std = visual_prompt.std().item()
                print(f"Sample {i+1} - Visual prompt stats: mean={mean:.4f}, std={std:.4f}")

                # pred = gpt2_tokenizer.decode(generated[0], skip_special_tokens=True)
                # print("Sample prediction completed.")

        gpt2_model.train()
        q_former.train()

if __name__ == "__main__":
    train()

    print("Completed sample predictions. Proceeding to evaluation...")

    # --- Evaluation on entire validation dataset ---
    from evaluate_metrics import evaluate_captions

    # Evaluate on the entire validation dataset
    print("\nStarting evaluation on validation set...")
    gpt2_model.eval()
    q_former.eval()
    all_preds = []
    all_refs = []
    for i, item in enumerate(eval_dataset):
        print(f"Processing sample {i+1}/{len(eval_dataset)}")
        image = item["image"]
        reference = item["caption"][0]

        pixel_values = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            vision_outputs = clip_model(pixel_values).last_hidden_state
            q_output = q_former(vision_outputs)
            q_output = q_output.to(dtype=torch.float32)

            start_token = gpt2_tokenizer.encode("A photo of", return_tensors="pt").to(device)
            start_embeds = gpt2_model.get_input_embeddings()(start_token).repeat(q_output.size(0), 1, 1)
            visual_prompt = torch.cat([start_embeds, q_output], dim=1)
            generated = gpt2_model.generate(
                inputs_embeds=visual_prompt,
                max_new_tokens=32,
                num_beams=3,
                pad_token_id=gpt2_tokenizer.eos_token_id,
                use_cache=True
            )
            pred = gpt2_tokenizer.decode(generated[0], skip_special_tokens=True)

        all_preds.append(pred)
        all_refs.append(reference)
        print(f"\nGT: {reference}\nPred: {pred}")

    metrics = evaluate_captions(all_preds, all_refs)
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    wandb.log(metrics)

    import random
    from torchvision.transforms.functional import to_pil_image
    samples_to_log = random.sample(list(eval_dataset), 3)
    for idx, item in enumerate(samples_to_log):
        image = item["image"]
        reference = item["caption"][0]

        pixel_values = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            vision_outputs = clip_model(pixel_values).last_hidden_state
            q_output = q_former(vision_outputs).to(dtype=torch.float32)
            start_token = gpt2_tokenizer.encode("A photo of", return_tensors="pt").to(device)
            start_embeds = gpt2_model.get_input_embeddings()(start_token).repeat(q_output.size(0), 1, 1)
            visual_prompt = torch.cat([start_embeds, q_output], dim=1)
            generated = gpt2_model.generate(
                inputs_embeds=visual_prompt,
                max_new_tokens=32,
                num_beams=3,
                pad_token_id=gpt2_tokenizer.eos_token_id,
                use_cache=True
            )
            pred = gpt2_tokenizer.decode(generated[0], skip_special_tokens=True)

        pil_img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()
        draw.text((5, 5), f"GT: {reference}", fill="white", font=font)
        draw.text((5, 25), f"Pred: {pred}", fill="white", font=font)

        wandb.log({f"Sample_{idx+1}": [wandb.Image(pil_img, caption=f"GT: {reference}\nPred: {pred}")]})

    torch.save({
        'q_former': q_former.state_dict(),
        'gpt2_model': gpt2_model.state_dict()
    }, "trained_model.pt")

    print("Uploading trained model to Weights & Biases...")
    final_model_artifact = wandb.Artifact("final-trained-model", type="model")
    final_model_artifact.add_file("trained_model.pt")
    wandb.log_artifact(final_model_artifact)
    print("Upload complete.")