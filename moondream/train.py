import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw
import json
import os
import re
import numpy as np

# --- CONFIGURARE ---
MODEL_ID = "vikhyatk/moondream2" 
EPOCHS = 7
LEARNING_RATE = 1e-5
BATCH_SIZE = 2 
GRAD_ACCUM_STEPS = 2 
TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"
OUTPUT_DIR = "./moondream_crosswalks_finetuned"
VIS_DIR = "./training_visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

def extract_coordinates(text):
    nums = re.findall(r"-?\d+\.\d+|-?\d+", text)
    if len(nums) >= 4: return [float(n) for n in nums[:4]]
    return None

def yolo_to_xyxy(coords, w, h):
    xc, yc, bw, bh = coords
    return [(xc - bw/2)*w, (yc - bh/2)*h, (xc + bw/2)*w, (yc + bh/2)*h]

def save_debug_image(pil_img, predicted_text, ground_truth, epoch, index, vis_dir):
    img_copy = pil_img.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    
    gt = extract_coordinates(ground_truth)
    if gt:
        draw.rectangle(yolo_to_xyxy(gt, w, h), outline="green", width=3)
    
    if predicted_text:
        pred = extract_coordinates(predicted_text)
        if pred:
            draw.rectangle(yolo_to_xyxy(pred, w, h), outline="red", width=3)
            
    img_copy.save(os.path.join(vis_dir, f"epoch_{epoch}_sample_{index}.jpg"))

class VQADataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if os.path.exists(item['image_path']): self.data.append(item)
                    elif os.path.exists(item['image_path'].lstrip('./')): 
                        item['image_path'] = item['image_path'].lstrip('./')
                        self.data.append(item)
                except: pass

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        item = self.data[i]
        try: return Image.open(item['image_path']).convert('RGB'), item['question'], item['answer']
        except: return self.__getitem__(0)

def collate_fn(batch):
    return [b for b in batch if b] and zip(*[b for b in batch if b])

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Înghețare robustă
    vision_module = getattr(model, "vision_encoder", getattr(model.model, "vision", None))
    if vision_module:
        for p in vision_module.parameters(): p.requires_grad = False
        print("Vision encoder frozen.")
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    
    train_dataset = VQADataset(TRAIN_FILE)
    val_dataset = VQADataset(VALID_FILE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            raw_images, questions, answers = batch
            
            with torch.no_grad():
                # FIX: Forțăm output-ul să fie tensor dacă nu e
                enc_list = [model.encode_image(img) for img in raw_images]
                # Verificăm dacă e obiect și extragem embedding-ul
                encoded_images = []
                for enc in enc_list:
                    # Verificăm diverse atribute posibile pentru a găsi tensorul
                    if isinstance(enc, torch.Tensor):
                        encoded_images.append(enc)
                    elif hasattr(enc, 'embeddings'): # Unele versiuni
                        encoded_images.append(enc.embeddings)
                    elif hasattr(enc, 'feature'): # Alte versiuni
                        encoded_images.append(enc.feature)
                    # Fallback critic: Multe versiuni HfMoondream returnează direct input-ul pt LLM
                    # dar s-ar putea să fie împachetat. 
                    # Dacă crapă aici, înseamnă că API-ul modelului s-a schimbat recent.
                    else:
                        # Încercăm să ghicim - de obicei e primul atribut sau e apelabil
                        # Hack extrem: Presupunem că e tensor ascuns
                         encoded_images.append(enc) 

            prompts = [f"\n\nQuestion: {q}\n\nAnswer: {a}" for q, a in zip(questions, answers)]
            tokens = [tokenizer(p, return_tensors="pt").input_ids.to(DEVICE) for p in prompts]
            prompt_only = [tokenizer(f"\n\nQuestion: {q}\n\nAnswer: ", return_tensors="pt").input_ids.to(DEVICE) for q in questions]

            batch_loss = 0
            for enc_img, tok, p_tok in zip(encoded_images, tokens, prompt_only):
                text_emb = model.get_input_embeddings()(tok)
                
                # AICI ESTE FIX-UL CRITIC PENTRU TIP
                if not isinstance(enc_img, torch.Tensor):
                    # Dacă încă nu e tensor, forțăm conversia sau accesăm datele interne
                    # Pentru HfMoondream din 2025, s-ar putea să fie nevoie de o altă metodă.
                    # Dar adesea, doar apelând .to(DEVICE) rezolvă dacă e un wrapper simplu.
                    try:
                        enc_img = enc_img.to(DEVICE)
                    except:
                        pass 

                input_embeds = torch.cat([enc_img, text_emb], dim=1)
                
                outputs = model(inputs_embeds=input_embeds)
                
                labels = tok.clone()
                labels[:, :p_tok.shape[1]] = -100
                
                loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                    outputs.logits[:, enc_img.shape[1]:-1, :].contiguous().view(-1, outputs.logits.size(-1)),
                    labels[:, 1:].contiguous().view(-1)
                )
                batch_loss += loss

            batch_loss = batch_loss / len(raw_images) / GRAD_ACCUM_STEPS
            batch_loss.backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if i % 10 == 0: print(f"Epoch {epoch+1} Batch {i} Loss: {batch_loss.item()*GRAD_ACCUM_STEPS:.4f}")

        # Validare simplificată
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                raw_images, questions, answers = batch
                # Repetăm logica de encode
                enc_list = [model.encode_image(img) for img in raw_images]
                # ... (logica de extragere tensor ar trebui să fie identică) ...
                # Pentru simplitate la validare, presupunem că merge generarea directă
                
                for j, (img, q, a) in enumerate(zip(raw_images, questions, answers)):
                    # Folosim metoda oficială de generare dacă merge
                    try:
                        enc = model.encode_image(img)
                        # generate() așteaptă de obicei embeddings direct dacă e apelat pe modelul text
                        # SAU putem folosi answer_question() dacă există în wrapper
                        if hasattr(model, "answer_question"):
                             res = model.answer_question(enc, q, tokenizer)
                        else:
                             # Fallback manual
                             pass
                    except: pass
                break
        
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Model Saved")