import os
import random
import numpy as np
import datasets
from datasets import concatenate_datasets, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from huggingface_hub import login
import torch
from pathlib import Path


# =========================================================================
# FUNCTII DE INCARCARE MODEL SI PROCESOR 
# =========================================================================

def fix_model_type_in_config_json(model_id: str):
    """Fix config.json by replacing 'lfm2-vl' model_type with 'lfm2_vl'."""
    import json
    from pathlib import Path

    config_path = Path(model_id) / "config.json"

    # Check if model_id is a local path
    with open(config_path, "r") as f:
        config = json.load(f)

    # Fix the model_type if needed
    if config.get("model_type") == "lfm2-vl":
        print(f"Fixing config.json for model {model_id}...")
        config["model_type"] = "lfm2_vl"

        # Write back the fixed config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print("config.json fixed successfully!")


def _download_and_cache_model(
    model_id: str, 
    hf_token: str, 
    processor_cache_path: Path, 
    model_weights_cache_path: Path
) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    """Helper function to download and cache model and processor."""
    
    # Download processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        max_image_tokens=256,
        token=hf_token,
    )
    
    # Download model
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="bfloat16",
        device_map="auto", 
        token=hf_token,
    )
    
    # Cache processor
    try:
        print(f"💾 Caching processor to {processor_cache_path}...")
        processor.save_pretrained(str(processor_cache_path))
        print("✅ Processor cached successfully")
    except Exception as e:
        print(f"⚠️ Failed to cache processor: {e}")
    
    # Cache model  
    try:
        print(f"💾 Caching model to {model_weights_cache_path}...")
        model.save_pretrained(str(model_weights_cache_path))
        
        # Apply config fix after caching
        try:
            fix_model_type_in_config_json(str(model_weights_cache_path))
        except Exception as e:
            print(f"Warning: could not fix config.json for model {model_id}: {e}")
            
        print("✅ Model cached successfully")
    except Exception as e:
        print(f"⚠️ Failed to cache model: {e}")
    
    return processor, model


def load_model_and_processor(
    model_id: str,
    cache_dir: str = "/models",
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Loads a model and processor from the Hugging Face model hub with Modal volume caching.
    """
    # Create cache directory structure
    model_cache_path = Path(cache_dir) / model_id.replace("/", "_")
    model_cache_path.mkdir(parents=True, exist_ok=True)
    
    processor_cache_path = model_cache_path / "processor"
    model_weights_cache_path = model_cache_path / "model"
    
    # Login using HF_TOKEN from environment variables
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("🔐 Logging in to Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("⚠️ No HF_TOKEN found in environment variables")

    # Check if model is cached
    if processor_cache_path.exists() and model_weights_cache_path.exists():
        print(f"📚 Loading cached model and processor from {model_cache_path}...")
        try:
            # Fix config if needed (for cached models)
            try:
                fix_model_type_in_config_json(str(model_weights_cache_path))
            except Exception as e:
                print(f"Warning: could not fix config.json for cached model: {e}")
            
            # Load processor from cache
            processor = AutoProcessor.from_pretrained(
                str(processor_cache_path),
                max_image_tokens=256,
                local_files_only=True,
            )
            
            # Load model from cache
            model = AutoModelForImageTextToText.from_pretrained(
                str(model_weights_cache_path),
                torch_dtype="bfloat16",
                device_map="auto",
                local_files_only=True,
            )
            
            print("✅ Successfully loaded model and processor from cache")
            
        except Exception as e:
            print(f"❌ Failed to load from cache: {e}")
            print(f"📚 Downloading model {model_id} from HuggingFace...")
            
            # Download and cache
            processor, model = _download_and_cache_model(model_id, hf_token, processor_cache_path, model_weights_cache_path)
    else:
        print(f"📚 Downloading model {model_id} from HuggingFace...")
        
        # Download and cache
        processor, model = _download_and_cache_model(model_id, hf_token, processor_cache_path, model_weights_cache_path)

    print("\n✅ Model loaded successfully!")
    print(f"📖 Vocab size: {len(processor.tokenizer)}")
    print(f"🔢 Parameters: {model.num_parameters():,}")
    print(f"💾 Model size: ~{model.num_parameters() * 2 / 1e9:.1f} GB (bfloat16)")

    return model, processor


# =========================================================================
# FUNCTIE CUSTOM LOAD_DATASET (pentru pietoni si semafoare)
# =========================================================================

def load_dataset(
    dataset_name: str, # Nu îl mai folosim direct, dar îl păstrăm pentru compatibilitate
    splits: list[str],
    n_samples: int | None = None,
    seed: int | None = 42,
    cache_dir: str = "/datasets",
) -> datasets.Dataset:
    
    # -------------------------------------------------------------------------
    # 1. Dataset Treceri de Pietoni
    # -------------------------------------------------------------------------
    print("🚶 Loading Crosswalk Dataset...")
    try:
        # Primul dataset
        ds_cross = datasets.load_dataset("zzd0225/crosswalk-detection-dataset", split="train")
    except Exception as e:
        print(f"⚠️ Primary crosswalk dataset failed ({e}). Using fallback...")
        # Fallback (dacă este necesar)
        ds_cross = datasets.load_dataset("keremberke/pedestrian-crossing-detection", "full", split="train")

    def format_cross(batch):
        texts = []
        # Iterăm pe baza numărului de imagini din lot pentru a asigura lungimea corectă
        batch_size = len(batch['image']) 
        
        for i in range(batch_size):
            text_response = "Nu văd o trecere de pietoni clară aici. Fii foarte atent!" # Răspuns implicit
            
            if 'objects' in batch and len(batch['objects']) > i:
                objs_data = batch['objects'][i] 
                categories = objs_data.get('category', [])
                
                if len(categories) > 0:
                    text_response = "Văd o trecere de pietoni marcată (zebră). Asigură-te că mașinile opresc înainte să traversezi."
            
            texts.append(text_response)

        return {'image': batch['image'], 'text_label': texts}

    # Aplicăm maparea doar dacă avem date
    if len(ds_cross) > 0:
        # Folosim o listă de coloane de eliminat
        cols_to_remove = [col for col in ds_cross.column_names if col not in ['image']]
        ds_cross = ds_cross.map(format_cross, batched=True, remove_columns=cols_to_remove)


    # -------------------------------------------------------------------------
    # 2. Dataset Semafoare (Adaptat pentru Pietoni)
    # -------------------------------------------------------------------------
    print("🚦 Loading Traffic Light Dataset...")
    
    # NEW DATASET: mehmetkeremturkcan/traffic-lights-of-new-york (include semafoare de pietoni)
    try:
        ds_lights = datasets.load_dataset("mehmetkeremturkcan/traffic-lights-of-new-york", split="train")
        print("✅ Folosind dataset-ul TLoNY (vehicular + pedestrian lights).")
    except Exception as e:
        print(f"⚠️ TLoNY dataset failed: {e}. Folosind un fallback simplu de clasificare.")
        
        # Fallback la un dataset mai robust, doar pentru culori (dacă TLoNY e blocat)
        try:
            ds_lights = datasets.load_dataset("OleFranz/TrafficLightDetectionAI", split="train")
        except Exception as e2:
             # Dacă și fallback-ul eșuează, e o problemă majoră.
             raise datasets.exceptions.DatasetNotFoundError(
                f"FATAL: Toate încercările de a găsi dataset de semafoare au eșuat. Eroarea finală: {e2}"
            )

    # Re-maparea bazată pe setul de date TLoNY sau OleFranz
    light_mapping = {
        # Mapping ajustat (valorile exacte depind de structura dataset-ului, dar acestea sunt bune ca default)
        0: "Semaforul indică ROȘU. Nu traversa! Așteaptă.",
        1: "Semaforul este GALBEN. Atenție sporită.",
        2: "Semaforul indică VERDE. Poți traversa.",
        3: "Semafor inactiv (sau altă clasă).",
    }

    def format_lights(batch):
        texts = []
        batch_size = len(batch['image'])
        
        # Iterăm pe baza dimensiunii lotului pentru a ne asigura că lista de ieșire are lungimea corectă
        for i in range(batch_size):
            text_response = "Situație necunoscută: Verifică semaforul și fii prudent."
            
            # Cazul Object Detection (TLoNY - preferat)
            if 'objects' in batch and len(batch['objects']) > i:
                objs_data = batch['objects'][i] 
                categories = objs_data.get('category', [])
                
                if len(categories) > 0:
                    # Folosim prima categorie detectată, cel mai probabil culoarea principală
                    cat_id = categories[0] 
                    text_response = light_mapping.get(cat_id, "Semafor detectat, dar starea este neclară. Fii atent!")
                else:
                    text_response = "Nu detectez niciun semafor activ în imagine."
            
            # Cazul Classification simplu (OleFranz - fallback)
            elif 'labels' in batch and len(batch['labels']) > i:
                label = batch['labels'][i]
                text_response = light_mapping.get(label, "Verifică semaforul.")
                
            texts.append(text_response) # Garantează lungimea corectă

        # --- DEBUG: VERIFICARE format_lights ---
        print("\n--- DEBUG: VERIFICARE format_lights ---")
        print(f"Dimensiunea lotului original (imagini): {len(batch['image'])}")
        print(f"Dimensiunea listei de texte returnate: {len(texts)}")
        print("Primele 2 etichete generate:", texts[:2])
        print("--------------------------------------\n")
        # -----------------------------
        
        return {'image': batch['image'], 'text_label': texts}

    # Aplicăm maparea
    cols_to_remove = [col for col in ds_lights.column_names if col not in ['image']]
    ds_lights = ds_lights.map(format_lights, batched=True, remove_columns=cols_to_remove)


    # -------------------------------------------------------------------------
    # 3. Combinare
    # -------------------------------------------------------------------------
    # Ne asigurăm că ambele dataset-uri au exact aceleași coloane: ['image', 'text_label']
    ds_cross = ds_cross.select_columns(['image', 'text_label'])
    ds_lights = ds_lights.select_columns(['image', 'text_label'])

    mixed_ds = concatenate_datasets([ds_cross, ds_lights])
    mixed_ds = mixed_ds.shuffle(seed=seed)

    if n_samples:
        n_samples = min(n_samples, len(mixed_ds))
        mixed_ds = mixed_ds.select(range(n_samples))

    print(f"✅ Pedestrian Assistant Dataset Ready: {len(mixed_ds)} samples.")
    return mixed_ds